import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eventwise.config import RetrievalConfig
from eventwise.embeddings import cosine_scores, normalise_rows
from eventwise.index import IndexStore, sync_index
from eventwise.retrieve import HybridRetriever, KeywordIndex, reciprocal_rank_fusion, top_indices
from retrieval_helpers import FakeEmbedder, figure, loaded_index, make_doc, matched, table, write_docx

BUDGET = "a" * 16
AGREEMENT = "b" * 16
TEMPLATE = "c" * 16
BOILERPLATE = "Payments are made within thirty days of a valid invoice being received by the finance team."


def _docs():
    budget = make_doc(BUDGET, "budget.pdf", title="Festival Budget", elements=[
        {"kind": "heading", "text": "3. Expenditure"},
        {"kind": "text", "text": "Security costs rose because extra night patrols were agreed; see Figure 2.",
         "section_path": ["3. Expenditure"]},
        {"kind": "table", "text": "expenditure", "label": "Table 3", "caption": "Table 3: Expenditure by line.",
         "caption_match": matched(), "section_path": ["3. Expenditure"], "bbox": (40, 100, 550, 400),
         "table": table([["Line", "Budget", "Actual"], ["Security & stewarding", "£165,000", "£176,400"],
                         ["Medical", "£55,000", "£58,300"]])},
        {"kind": "figure", "text": "Figure 2: Largest expenditure lines.", "label": "Figure 2",
         "caption": "Figure 2: Largest expenditure lines.", "caption_match": matched(), "figure": figure(),
         "section_path": ["3. Expenditure"], "page": 2},
        {"kind": "text", "text": BOILERPLATE, "section_path": ["4. Terms"], "page": 2},
    ])
    agreement = make_doc(AGREEMENT, "Agreement_CW-AGR-2026-003.pdf", elements=[
        {"kind": "text", "text": "Agreement CW-AGR-2026-003 with Cambrian Event Security. Total contract value £176,400."},
        {"kind": "text", "text": BOILERPLATE},
    ])
    template = make_doc(TEMPLATE, "Agreement_Template.docx", elements=[
        {"kind": "text", "text": BOILERPLATE},
        {"kind": "text", "text": "Catering is provided for crew on shift."},
    ])
    return [budget, agreement, template]


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(loaded_index(_docs(), FakeEmbedder()), FakeEmbedder())


# ---------------------------------------------------------------- primitives
def test_normalised_vectors_and_cosine():
    v = normalise_rows(np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(v[0], [0.6, 0.8]) and np.allclose(v[1], [0, 0])  # zero rows stay zero, no NaN
    scores = cosine_scores(np.array([1.0, 0.0], dtype=np.float32), v)
    assert np.allclose(scores, [0.6, 0.0, 1.0])


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion({"bm25": [1, 2, 3], "dense": [3, 1]}, k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert sorted(fused, key=lambda i: -fused[i]) == [1, 3, 2]
    weighted = reciprocal_rank_fusion({"bm25": [1], "dense": [2]}, {"bm25": 2.0, "dense": 1.0}, k=60)
    assert weighted[1] > weighted[2]


def test_top_indices_respects_mask_and_threshold():
    scores = np.array([0.5, 0.9, 0.0, 0.7])
    mask = np.array([True, False, True, True])
    assert top_indices(scores, mask, 10) == [3, 0, 2]
    assert top_indices(scores, mask, 10, min_score=0.0) == [3, 0]


def test_bm25_exact_identifier_and_number():
    texts = ["Agreement CW-AGR-2026-003 total £176,400", "Agreement CW-AGR-2026-004 total £90,000", "Unrelated"]
    kw = KeywordIndex(texts)
    assert int(np.argmax(kw.scores("CW-AGR-2026-003"))) == 0
    assert int(np.argmax(kw.scores("176400"))) == 0
    assert kw.scores("").sum() == 0 and KeywordIndex([]).scores("x").size == 0


# ---------------------------------------------------------------- retrieval
def test_hybrid_returns_source_metadata(retriever):
    resp = retriever.search("security & stewarding actual spend", top_k=5)
    top = resp.results[0]
    assert top.element_id == f"{BUDGET[:10]}:p1:tab2" and top.kind == "table"
    assert top.doc_name == "budget.pdf" and top.page == 1 and top.label == "Table 3"
    assert top.caption == "Table 3: Expenditure by line." and top.caption_confidence == "high"
    assert top.section_path == ["3. Expenditure"] and "§ 3. Expenditure" in top.citation
    assert top.element_ref["element_id"] == top.element_id
    assert top.element_ref["document_json"].endswith(f"{BUDGET}/document.json")
    assert top.bm25_rank and top.dense_rank and top.score > 0
    assert set(resp.timings_ms) >= {"bm25", "dense", "query_embedding", "total"}


def test_figure_result_has_image_path_and_citing_context(retriever):
    resp = retriever.search("largest expenditure lines chart", kinds=["figure"])
    fig = resp.results[0]
    assert fig.kind == "figure" and fig.image_path == str(Path("/tmp/extracted") / BUDGET / "images/p1_fig1.png")
    relations = {(c.relation, c.kind) for c in fig.context}
    assert ("cited_by", "text") in relations  # the paragraph that says "see Figure 2"


def test_document_filter(retriever):
    resp = retriever.search("security contract value", documents=["Agreement_CW"])
    assert resp.results and {r.doc_id for r in resp.results} == {AGREEMENT}
    resp = retriever.search("security", documents=[BUDGET])
    assert {r.doc_id for r in resp.results} == {BUDGET}
    resp = retriever.search("security", documents=["no-such-doc"])
    assert resp.results == [] and any("matches no indexed document" in w for w in resp.warnings)


def test_kind_filter(retriever):
    for kind in ("table", "text", "heading", "figure"):
        resp = retriever.search("expenditure security", kinds=[kind])
        assert resp.results and {r.kind for r in resp.results} == {kind}
    with pytest.raises(ValueError):
        retriever.search("x", kinds=["paragraph"])


def test_duplicate_evidence_listed_once(retriever):
    resp = retriever.search("payments within thirty days of a valid invoice", top_k=10)
    hits = [r for r in resp.results if r.content == BOILERPLATE]
    assert len(hits) == 1
    assert {a["doc_name"] for a in hits[0].also_in} == {"budget.pdf", "Agreement_CW-AGR-2026-003.pdf",
                                                         "Agreement_Template.docx"} - {hits[0].doc_name}
    assert sum(s["reason"] == "duplicate" for s in resp.suppressed) == 2


def test_row_group_hit_collapses_to_one_table_result():
    rows = [["Name", "Role", "Phone"]] + [[f"Person {i}", f"role{i}", f"0700 {i:04d}"] for i in range(30)]
    doc = make_doc("d" * 16, "crew.pdf", [{"kind": "table", "text": "crew", "label": "Table 1",
                                           "caption": "Table 1: Crew list.", "table": table(rows)}])
    r = HybridRetriever(loaded_index([doc], FakeEmbedder()), FakeEmbedder())
    resp = r.search("Person 17 role17 phone", top_k=5)
    assert len(resp.results) == 1
    res = resp.results[0]
    assert "Person 17" in res.content and res.matched_parts and res.matched_parts[0]["rows"][0] <= 18
    assert "rows" in res.citation
    full = res.element_ref["element_id"]
    assert full == doc["elements"][0]["element_id"]


def test_bm25_only_mode_without_embedder():
    r = HybridRetriever(loaded_index(_docs(), FakeEmbedder()), embedder=None)
    resp = r.search("CW-AGR-2026-003")
    assert resp.mode == "bm25" and resp.results[0].doc_id == AGREEMENT
    assert any("BM25 only" in w for w in resp.warnings)


def test_embedder_mismatch_disables_dense():
    r = HybridRetriever(loaded_index(_docs(), FakeEmbedder()), FakeEmbedder(name="fake:other"))
    assert r.embedder is None and any("differs from index embedder" in w for w in r.warnings)


def test_per_method_guarantee_keeps_strong_single_method_hit():
    # Unit 8 is BM25's top hit but absent from the dense list; plain RRF ranks it below the top 3.
    r = HybridRetriever(loaded_index(_docs(), FakeEmbedder()), FakeEmbedder())
    fused = reciprocal_rank_fusion({"bm25": [8, 1, 2, 3], "dense": [1, 2, 3, 5, 0]})
    assert sorted(fused, key=lambda i: -fused[i])[:3] == [1, 2, 3]
    unit_rank = {"bm25": {8: 1, 1: 2, 2: 3, 3: 4}, "dense": {1: 1, 2: 2, 3: 3, 5: 4, 0: 5}}
    zeros = np.zeros(len(r.units))
    cfg = replace(RetrievalConfig(), same_document_decay=1.0)
    target = r.units[8].element_id
    plain, _ = r._select(dict(fused), unit_rank, zeros, zeros, 3, cfg, guaranteed=[])
    kept, suppressed = r._select(dict(fused), unit_rank, zeros, zeros, 3, cfg,
                                 guaranteed=[(target, "top 1 of bm25")])
    assert target not in {x.element_id for x in plain}
    assert [x.element_id for x in kept][:2] == [x.element_id for x in plain][:2]  # higher ranks untouched
    assert kept[-1].element_id == target and kept[-1].selection.startswith("guaranteed") and len(kept) == 3
    assert any(s["reason"].startswith("displaced") for s in suppressed)


def test_guarantee_never_displaces_for_a_rejected_item():
    r = HybridRetriever(loaded_index(_docs(), FakeEmbedder()), FakeEmbedder())
    fused = reciprocal_rank_fusion({"bm25": [1, 2, 3, 0]})  # unit 0 = heading of the section already shown
    unit_rank = {"bm25": {1: 1, 2: 2, 3: 3, 0: 4}}
    zeros = np.zeros(len(r.units))
    kept, _ = r._select(dict(fused), unit_rank, zeros, zeros, 3, RetrievalConfig(),
                        guaranteed=[(r.units[0].element_id, "test")])
    assert len(kept) == 3 and r.units[0].element_id not in {x.element_id for x in kept}


def test_guarantee_scales_with_top_k(retriever):
    resp = retriever.search("CW-AGR-2026-003 Cambrian contract value", documents=[AGREEMENT], top_k=1)
    assert resp.results[0].selection == "fused"  # k=1: no guaranteed slots, the fused winner stays


def test_same_document_decay_is_soft(retriever):
    strict = retriever.search("security expenditure agreement", top_k=6,
                              config=replace(RetrievalConfig(), max_per_document=1))
    assert len({r.doc_id for r in strict.results}) == len(strict.results)  # hard cap: one per document
    soft = retriever.search("security expenditure agreement", top_k=6)
    per_doc = {}
    for res in soft.results:
        per_doc[res.doc_id] = per_doc.get(res.doc_id, 0) + 1
    assert max(per_doc.values()) > 1  # decay still allows several results from one document
    assert AGREEMENT in per_doc
    fused = [r.selection_score for r in soft.results if r.selection == "fused"]
    assert fused == sorted(fused, reverse=True)  # order follows the decay-adjusted score


def test_neighbour_expansion_adjacent_elements(retriever):
    resp = retriever.search("CW-AGR-2026-003 Cambrian contract value", documents=[AGREEMENT], top_k=1)
    ctx = resp.results[0].context
    assert ctx and ctx[0].relation == "next"
    off = retriever.search("CW-AGR-2026-003", top_k=1, config=replace(RetrievalConfig(), expand_neighbours=False))
    assert off.results[0].context == []


def test_get_element_returns_full_original(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_docx(src / "a.docx", "Alpha", ["Mileage is paid at 45p per mile."])
    sync_index([src], FakeEmbedder(), tmp_path / "index", tmp_path / "extracted")
    r = HybridRetriever.from_disk(tmp_path / "index", FakeEmbedder())
    res = r.search("mileage 45p").results[0]
    original = r.get_element(res.element_id)
    doc = json.loads(Path(res.element_ref["document_json"]).read_text())
    assert original == next(e for e in doc["elements"] if e["element_id"] == res.element_id)
    assert "45p" in original["text"] and res.location == original["location"]


def test_config_validation():
    assert RetrievalConfig.from_dict({"top_k": 3}).top_k == 3
    with pytest.raises(ValueError, match="Unknown"):
        RetrievalConfig.from_dict({"topk": 3})


# ---------------------------------------------------------------- real model / real corpus
def _model_available() -> bool:
    return any(Path(".cache/models").glob("*all-MiniLM-L6-v2*"))


@pytest.mark.skipif(not _model_available(), reason="all-MiniLM-L6-v2 not downloaded")
def test_real_model_dimensions_and_similarity():
    from eventwise.embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(allow_download=False)
    v = emb.encode(["mileage rate per mile for car journeys", "reimbursement for driving a personal car",
                    "the festival headline act on Saturday"])
    assert v.shape == (3, 384) and emb.dim == 384 and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert emb.encode([]).shape == (0, 384)


@pytest.mark.skipif(not Path(".cache/index/manifest.json").exists(), reason="corpus index not built")
def test_corpus_index_identifier_lookup():
    r = HybridRetriever(IndexStore(".cache/index").load(), embedder=None)
    resp = r.search("CW-AGR-2026-003", top_k=10, mode="bm25")
    names = {x.doc_name for x in resp.results}
    assert "Agreement_CW-AGR-2026-003.pdf" in names
    assert {"Invoice_CES-26-0388.pdf", "Invoice_CES-26-0431.pdf"} & names
