"""Upload and background indexing against temporary directories. No OpenAI calls."""

import threading
import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from eventwise.indexing.index import IndexStore, SyncReport, sync_index
from eventwise.ingestion.service import IngestionError
from eventwise.retrieval.retrieve import HybridRetriever
from retrieval_helpers import FakeEmbedder, write_docx

from eventwise.api.app import create_app


def pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def docx_bytes(path, heading: str, paragraphs: list[str]) -> bytes:
    write_docx(path, heading, paragraphs)
    return path.read_bytes()


def wait_until(client, predicate, timeout: float = 10):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get("/api/ingestion/status").json()
        if predicate(last):
            return last
        time.sleep(0.02)
    raise AssertionError(last)


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr("eventwise.api.app.answer_question", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("ask should not call the model in this test")
    ))
    data = tmp_path / "data"
    index = tmp_path / "index"
    extracted = tmp_path / "extracted"
    data.mkdir()
    original = data / "original-documents" / "old.docx"
    original.parent.mkdir(parents=True)
    docx_bytes(original, "Old", ["The original note stays in the knowledge base."])
    embedder = FakeEmbedder()
    sync_index([data], embedder, index, extracted)
    before = IndexStore(index).read_manifest()

    def loader():
        return HybridRetriever(IndexStore(index).load(), embedder), object()

    app = create_app(index_dir=index, data_dir=data, extracted_dir=extracted, loader=loader)
    with TestClient(app) as client:
        yield client, data, index, before


def test_uploads_one_pdf_and_publishes_a_searchable_index(library):
    client, data, index, before = library
    previous = client.app.state.retriever
    response = client.post("/api/documents", files=[
        ("files", ("reports/notes.pdf", pdf_bytes("A new safety note."), "application/pdf")),
    ])
    assert response.status_code == 202
    assert response.json()["saved"] == ["reports/notes.pdf"]
    status = wait_until(client, lambda item: item["state"] == "ready")
    assert status["previous_available"] is True
    assert status["documents"] == 2
    assert status["units"] >= before["documents"][next(iter(before["documents"]))]["n_units"]
    assert (data / "reports" / "notes.pdf").is_file()
    assert (data / "original-documents" / "old.docx").is_file()
    names = {item["name"] for item in client.get("/api/documents").json()["documents"]}
    assert names == {"old.docx", "notes.pdf"}
    assert client.app.state.retriever is not previous
    assert len(client.app.state.retriever.index.documents) == 2
    assert not (index.parent / "index-staging").exists()


def test_uploads_multiple_files_and_a_nested_folder(library):
    client, _data, _index, _before = library
    folder = docx_bytes(library[1] / "_draft.docx", "Draft", ["Nested contract text."])
    (library[1] / "_draft.docx").unlink()
    response = client.post("/api/documents", files=[
        ("files", ("festival-2026/contracts.pdf", pdf_bytes("Contract terms."), "application/pdf")),
        ("files", ("festival-2026/invoices/june.docx", folder, "application/octet-stream")),
    ])
    assert response.status_code == 202
    assert response.json()["saved"] == ["festival-2026/contracts.pdf", "festival-2026/invoices/june.docx"]
    wait_until(client, lambda item: item["state"] == "ready")
    paths = {item["path"] for item in client.get("/api/documents").json()["documents"]}
    assert any(path.endswith("festival-2026/contracts.pdf") for path in paths)
    assert any(path.endswith("festival-2026/invoices/june.docx") for path in paths)
    assert any(path.endswith("original-documents/old.docx") for path in paths)


def test_rejects_unsupported_unsafe_and_duplicate_files(library):
    client, data, index, before = library
    original = (data / "original-documents" / "old.docx").read_bytes()
    unsupported = client.post("/api/documents", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    unsafe = client.post("/api/documents", files=[("files", ("../outside.pdf", pdf_bytes("nope"), "application/pdf"))])
    nested = client.post("/api/documents", files=[("files", ("folder/../../outside.pdf", pdf_bytes("nope"), "application/pdf"))])
    duplicate = client.post("/api/documents", files=[
        ("files", ("original-documents/old.docx", docx_bytes(data / "_other.docx", "X", ["y"]), "application/octet-stream")),
    ])
    (data / "_other.docx").unlink(missing_ok=True)
    assert unsupported.status_code == 400
    assert unsafe.status_code == 400
    assert nested.status_code == 400
    assert duplicate.status_code == 409
    assert (data / "original-documents" / "old.docx").read_bytes() == original
    assert IndexStore(index).read_manifest()["documents"] == before["documents"]
    assert list(data.rglob("*.pdf")) == []
    assert client.get("/api/ingestion/status").json()["state"] == "ready"


def test_rejects_a_second_upload_and_questions_while_processing(library, monkeypatch):
    client, _data, _index, _before = library
    started = threading.Event()
    release = threading.Event()

    def pause(*args, **kwargs):
        started.set()
        assert release.wait(5)
        raise RuntimeError("stopped after the pause")

    monkeypatch.setattr("eventwise.ingestion.service.sync_index", pause)
    response = client.post("/api/documents", files=[
        ("files", ("fresh.pdf", pdf_bytes("Wait."), "application/pdf")),
    ])
    assert response.status_code == 202
    assert started.wait(5)
    status = client.get("/api/ingestion/status").json()
    assert status["state"] == "processing"
    assert status["files_processing"] == 1
    assert status["previous_available"] is True
    again = client.post("/api/documents", files=[
        ("files", ("other.pdf", pdf_bytes("Later."), "application/pdf")),
    ])
    assert again.status_code == 409
    assert not (_data / "other.pdf").exists()
    asking = client.post("/api/ask", json={"question": "What remains?"})
    assert asking.status_code == 503
    assert "Ready" in asking.json()["detail"]
    release.set()
    failed = wait_until(client, lambda item: item["state"] == "failed")
    assert failed["previous_available"] is True
    assert "stopped after the pause" in failed["error"]
    assert client.app.state.retriever.index.documents == _before["documents"]


def test_failed_indexing_keeps_the_previous_index(library, monkeypatch):
    client, data, index, before = library
    manifest_before = (index / "manifest.json").read_bytes()

    def fail(*args, **kwargs):
        report = SyncReport()
        report.failed.append(("fresh.pdf", "parse failed"))
        return report

    monkeypatch.setattr("eventwise.ingestion.service.sync_index", fail)
    response = client.post("/api/documents", files=[
        ("files", ("fresh.pdf", pdf_bytes("Broken later."), "application/pdf")),
    ])
    assert response.status_code == 202
    status = wait_until(client, lambda item: item["state"] == "failed")
    assert status["previous_available"] is True
    assert "parse failed" in status["error"]
    assert (index / "manifest.json").read_bytes() == manifest_before
    assert (data / "original-documents" / "old.docx").is_file()
    assert not (data / "fresh.pdf").exists()
    assert not (index.parent / "index-staging").exists()
    listed = client.get("/api/documents").json()["documents"]
    assert [item["name"] for item in listed] == ["old.docx"]


def test_questions_are_rejected_when_no_index_is_loaded(tmp_path):
    from eventwise.indexing.index import IndexUnavailable

    def loader():
        raise IndexUnavailable("No index at .cache/index.")

    app = create_app(loader=loader, data_dir=tmp_path / "data", index_dir=tmp_path / "index",
                      extracted_dir=tmp_path / "extracted")
    with TestClient(app) as client:
        status = client.get("/api/ingestion/status")
        asking = client.post("/api/ask", json={"question": "Anything?"})
    assert status.status_code == 200
    assert status.json()["state"] == "not_ready"
    assert status.json()["previous_available"] is False
    assert asking.status_code == 503


def test_upload_limit_is_enforced(tmp_path):
    from eventwise.ingestion.service import IngestionService, UploadLimits

    service = IngestionService(tmp_path / "data", tmp_path / "index", tmp_path / "extracted",
                               limits=UploadLimits(max_file_bytes=8, max_total_bytes=20))
    with pytest.raises(IngestionError) as caught:
        service.save_uploads([("tiny.pdf", Bytes())])
    assert caught.value.status_code == 413
    assert list((tmp_path / "data").rglob("*")) == []


class Bytes:
    def read(self, _n):
        return b"%PDF-too-big"
