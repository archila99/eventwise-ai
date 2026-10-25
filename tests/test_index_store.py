import json

import numpy as np
import pytest

from eventwise.config import IndexConfig
from eventwise.indexing.index import IndexStore, IndexUnavailable, sync_index
from retrieval_helpers import FakeEmbedder, write_docx


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    write_docx(src / "alpha.docx", "Alpha policy", ["Mileage is paid at 45p per mile.", "Claims need receipts."])
    write_docx(src / "beta.docx", "Beta guide", ["Drones are prohibited on site.", "Glass is not allowed."])
    return src, tmp_path / "index", tmp_path / "extracted"


def test_build_persist_and_reload(corpus):
    src, index_dir, extracted = corpus
    emb = FakeEmbedder()
    report = sync_index([src], emb, index_dir, extracted)
    assert sorted(report.added) == ["alpha.docx", "beta.docx"] and report.extracted == 2
    loaded = IndexStore(index_dir).load()
    assert len(loaded.units) == report.units_total == loaded.vectors.shape[0]
    assert loaded.vectors.shape[1] == emb.dim and loaded.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0, atol=1e-5)
    # vector row i belongs to unit i: re-encoding the unit text reproduces the stored row
    again = emb.encode([u.search_text for u in loaded.units])
    assert np.allclose(again, loaded.vectors, atol=1e-6)
    assert loaded.embedder_name == emb.name and not loaded.problems


def test_unchanged_documents_reuse_cache_without_embedding(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    emb = FakeEmbedder()
    report = sync_index([src], emb, index_dir, extracted)
    assert sorted(report.reused) == ["alpha.docx", "beta.docx"]
    assert report.units_embedded == 0 and emb.calls == 0 and report.extracted == 0


def test_incremental_add_embeds_only_the_new_document(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    write_docx(src / "gamma.docx", "Gamma", ["Invoice CES-26-0388 was paid."])
    emb = FakeEmbedder()
    report = sync_index([src], emb, index_dir, extracted)
    assert report.added == ["gamma.docx"] and len(report.reused) == 2
    gamma_units = [u for u in IndexStore(index_dir).load().units if u.doc_name == "gamma.docx"]
    assert report.units_embedded == emb.texts_encoded == len(gamma_units) > 0


def test_changed_and_removed_documents_are_detected(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    old_ids = set(IndexStore(index_dir).read_manifest()["documents"])
    write_docx(src / "alpha.docx", "Alpha policy", ["Mileage is now paid at 55p per mile."])
    (src / "beta.docx").unlink()
    report = sync_index([src], FakeEmbedder(), index_dir, extracted)
    assert report.changed == ["alpha.docx"] and report.removed == ["beta.docx"]
    manifest = IndexStore(index_dir).read_manifest()
    assert len(manifest["documents"]) == 1 and not (set(manifest["documents"]) & old_ids)
    for doc_id in old_ids:  # stale files are deleted, not just forgotten
        assert not IndexStore(index_dir).doc_dir(doc_id).exists()
    texts = " ".join(u.content for u in IndexStore(index_dir).load().units)
    assert "55p" in texts and "45p" not in texts and "Drones" not in texts


def test_removed_even_when_not_in_scanned_paths(corpus, tmp_path):
    src, index_dir, extracted = corpus
    other = tmp_path / "other"
    other.mkdir()
    write_docx(other / "delta.docx", "Delta", ["Delta content."])
    sync_index([src, other], FakeEmbedder(), index_dir, extracted)
    (other / "delta.docx").unlink()
    report = sync_index([src], FakeEmbedder(), index_dir, extracted)  # other/ no longer scanned
    assert report.removed == ["delta.docx"]


def test_documents_outside_scan_are_kept_unless_pruned(corpus, tmp_path):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    only_alpha = [src / "alpha.docx"]
    report = sync_index(only_alpha, FakeEmbedder(), index_dir, extracted)
    assert report.removed == [] and len(IndexStore(index_dir).read_manifest()["documents"]) == 2
    report = sync_index(only_alpha, FakeEmbedder(), index_dir, extracted, prune=True)
    assert report.removed == ["beta.docx"] and len(IndexStore(index_dir).read_manifest()["documents"]) == 1


def test_cache_invalidated_by_embedder_or_settings(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    report = sync_index([src], FakeEmbedder(dim=32, name="fake:bow-32"), index_dir, extracted)
    assert sorted(report.reembedded) == ["alpha.docx", "beta.docx"] and report.extracted == 0
    assert IndexStore(index_dir).load().vectors.shape[1] == 32
    report = sync_index([src], FakeEmbedder(dim=32, name="fake:bow-32"), index_dir, extracted,
                        config=IndexConfig(context_words=5))
    assert sorted(report.reembedded) == ["alpha.docx", "beta.docx"]  # unit construction changed
    report = sync_index([src], FakeEmbedder(dim=32, name="fake:bow-32"), index_dir, extracted, rebuild=True)
    assert report.units_embedded == report.units_total


def test_missing_index_raises_clear_error(tmp_path):
    with pytest.raises(IndexUnavailable, match="No index"):
        IndexStore(tmp_path / "nothing").load()


def test_corrupt_manifest_is_reported_and_rebuilt(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    (index_dir / "manifest.json").write_text("{not json")
    with pytest.raises(IndexUnavailable, match="corrupt"):
        IndexStore(index_dir).load()
    report = sync_index([src], FakeEmbedder(), index_dir, extracted)
    assert any("corrupt" in n for n in report.notes) and len(report.added) == 2
    assert len(IndexStore(index_dir).load().units) == report.units_total


@pytest.mark.parametrize("damage", ["truncate_vectors", "wrong_shape", "delete_units", "garbage_units"])
def test_corrupt_document_files_are_skipped_then_repaired(corpus, damage):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    store = IndexStore(index_dir)
    doc_id, info = next((d, i) for d, i in store.read_manifest()["documents"].items() if i["doc_name"] == "alpha.docx")
    target = store.doc_dir(doc_id)
    if damage == "truncate_vectors":
        (target / "vectors.npy").write_bytes((target / "vectors.npy").read_bytes()[:40])
    elif damage == "wrong_shape":
        np.save(target / "vectors.npy", np.zeros((1, 3), dtype=np.float32))
    elif damage == "delete_units":
        (target / "units.json").unlink()
    else:
        (target / "units.json").write_text("[{]")
    loaded = store.load()
    assert len(loaded.problems) == 1 and "alpha.docx" in loaded.problems[0]
    assert {u.doc_name for u in loaded.units} == {"beta.docx"}  # the healthy document still loads
    emb = FakeEmbedder()
    report = sync_index([src], emb, index_dir, extracted)
    repaired = store.load()
    assert not repaired.problems and {u.doc_name for u in repaired.units} == {"alpha.docx", "beta.docx"}
    assert report.reembedded == ["alpha.docx"] and report.reused == ["beta.docx"]
    assert emb.texts_encoded == sum(u.doc_name == "alpha.docx" for u in repaired.units)


def test_corrupt_extraction_cache_is_re_extracted(corpus):
    src, index_dir, extracted = corpus
    sync_index([src], FakeEmbedder(), index_dir, extracted)
    for f in extracted.glob("*/document.json"):
        f.write_text("broken")
    report = sync_index([src], FakeEmbedder(), index_dir, extracted)
    assert report.extracted == 2 and len(report.reused) == 2  # same content -> same units -> vectors reused
    assert all(json.loads(f.read_text())["doc_id"] for f in extracted.glob("*/document.json"))


def test_unreadable_source_reported_not_raised(corpus):
    src, index_dir, extracted = corpus
    (src / "broken.docx").write_bytes(b"not a zip")
    report = sync_index([src], FakeEmbedder(), index_dir, extracted)
    assert len(report.failed) == 1 and report.failed[0][0].endswith("broken.docx")
    assert len(report.added) == 2
    assert "broken.docx" not in {i["doc_name"] for i in IndexStore(index_dir).read_manifest()["documents"].values()}
