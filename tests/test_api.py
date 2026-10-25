"""API tests with a fake index and a mocked answer call. No OpenAI requests."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from eventwise.answer import Citation, GroundedAnswer
from eventwise.api.app import create_app
from eventwise.indexing.index import IndexUnavailable


class Unit:
    def __init__(self, **kwargs):
        self.doc_name = kwargs.get("doc_name", "alpha.pdf")
        self.page = kwargs.get("page", 3)
        self.page_label = kwargs.get("page_label", "3")
        self.section_path = kwargs.get("section_path", ["4. Safety and welfare"])
        self.label = kwargs.get("label", "Table 1")
        self.kind = kwargs.get("kind", "table")


def retriever():
    return SimpleNamespace(
        index=SimpleNamespace(documents={
            "a": {"doc_name": "beta.docx"},
            "b": {"doc_name": "alpha.pdf"},
            "c": {"doc_name": "alpha.pdf"},
        }),
        primary={"e1": 0},
        units=[Unit()],
    )


def answer(query="What was the security spend?", citations=None, sufficient=True):
    return GroundedAnswer(
        query=query,
        answer="The security spend was £176,400.",
        sufficient=sufficient,
        citations=citations if citations is not None else [
            Citation(element_id="e1", citation="alpha.pdf — p. 3 — § 4. Safety and welfare — Table 1", kind="table"),
        ],
    )


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_answer(used_retriever, query, client, **kwargs):
        seen.append({"query": query, "client": client, "kwargs": kwargs, "retriever": used_retriever})
        return answer(query)

    monkeypatch.setattr("eventwise.api.app.answer_question", fake_answer)
    return seen


@pytest.fixture
def client(calls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    app = create_app(loader=lambda: (retriever(), object()))
    with TestClient(app) as test_client:
        yield test_client


def test_lists_unique_document_names(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [
        {"id": "b", "name": "alpha.pdf", "path": ""},
        {"id": "c", "name": "alpha.pdf", "path": ""},
        {"id": "a", "name": "beta.docx", "path": ""},
    ]}


def test_answers_across_all_documents(client, calls):
    response = client.post("/api/ask", json={"question": "  What was the security spend?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The security spend was £176,400."
    assert body["sufficient"] is True
    assert body["citations"] == [{
        "element_id": "e1",
        "citation": "alpha.pdf — p. 3 — § 4. Safety and welfare — Table 1",
        "kind": "table",
        "document": "alpha.pdf",
        "page": 3,
        "page_label": "3",
        "section": "4. Safety and welfare",
        "label": "Table 1",
    }]
    assert calls[0]["query"] == "What was the security spend?"
    assert calls[0]["kwargs"]["documents"] is None
    assert calls[0]["kwargs"].get("top_k") is None
    assert calls[0]["kwargs"].get("model") is None


def test_filters_by_document_name(client, calls):
    response = client.post("/api/ask", json={"question": "Who played Friday?", "document": "beta.docx"})
    assert response.status_code == 200
    assert calls[0]["kwargs"]["documents"] == ["a"]
    assert calls[0]["client"] is not None


def test_duplicate_document_names_must_be_chosen_exactly(client, calls):
    response = client.post("/api/ask", json={"question": "Who played Friday?", "document": "alpha.pdf"})
    assert response.status_code == 409
    chosen = client.post("/api/ask", json={"question": "Who played Friday?", "document": "b"})
    assert chosen.status_code == 200
    assert calls[0]["kwargs"]["documents"] == ["b"]


def test_rejects_an_empty_question(client, calls):
    for question in ("", "   "):
        response = client.post("/api/ask", json={"question": question})
        assert response.status_code == 400
        assert response.json()["detail"] == "Enter a question."
    assert calls == []


def test_rejects_a_missing_question(client):
    response = client.post("/api/ask", json={})
    assert response.status_code == 422


def test_rejects_an_unknown_document(client, calls):
    response = client.post("/api/ask", json={"question": "Who played Friday?", "document": "missing.pdf"})
    assert response.status_code == 404
    assert "missing.pdf" in response.json()["detail"]
    assert calls == []


def test_missing_index_is_unavailable():
    def loader():
        raise IndexUnavailable("No index at .cache/index. Build it with: python -m eventwise.indexing data/")

    app = create_app(loader=loader)
    with TestClient(app) as test_client:
        documents = test_client.get("/api/documents")
        ask = test_client.post("/api/ask", json={"question": "Who played Friday?"})
    assert documents.status_code == 503
    assert "No index" in documents.json()["detail"]
    assert ask.status_code == 503


def test_missing_api_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(loader=lambda: (retriever(), object()))
    with TestClient(app) as test_client:
        documents = test_client.get("/api/documents")
        ask = test_client.post("/api/ask", json={"question": "Who played Friday?"})
    assert documents.status_code == 200
    assert ask.status_code == 503
    assert "OPENAI_API_KEY" in ask.json()["detail"]


def test_model_failure_is_a_bad_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    def fail(*args, **kwargs):
        raise RuntimeError("upstream failed")

    monkeypatch.setattr("eventwise.api.app.answer_question", fail)
    app = create_app(loader=lambda: (retriever(), object()))
    with TestClient(app) as test_client:
        response = test_client.post("/api/ask", json={"question": "Who played Friday?"})
    assert response.status_code == 502
    assert response.json()["detail"] == "The model request failed. Try again in a moment."
    assert "upstream failed" not in response.text
    assert "sk-test" not in response.text


def test_citation_without_an_index_unit_keeps_the_pipeline_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    def fake_answer(*args, **kwargs):
        return answer(citations=[Citation(element_id="missing", citation="notes.docx — paragraph 4", kind="text")])

    monkeypatch.setattr("eventwise.api.app.answer_question", fake_answer)
    bare = SimpleNamespace(index=SimpleNamespace(documents={"a": {"doc_name": "notes.docx"}}), primary={}, units=[])
    app = create_app(loader=lambda: (bare, object()))
    with TestClient(app) as test_client:
        response = test_client.post("/api/ask", json={"question": "Where is it?"})
    assert response.status_code == 200
    citation = response.json()["citations"][0]
    assert citation["element_id"] == "missing"
    assert citation["citation"] == "notes.docx — paragraph 4"
    assert citation["kind"] == "text"
    assert citation["document"] is None
    assert citation["page"] is None
    assert citation["section"] is None
    assert citation["label"] is None


def test_frontend_is_served_without_html_injection():
    page = (create_app(loader=lambda: (retriever(), object())))
    with TestClient(page) as test_client:
        response = test_client.get("/")
        script = test_client.get("/static/app.js")
    assert response.status_code == 200
    assert "EventWise AI" in response.text
    assert script.status_code == 200
    assert "innerHTML" not in script.text
    assert "insertAdjacentHTML" not in script.text
    assert "textContent" in script.text
