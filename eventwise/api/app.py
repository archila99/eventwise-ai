"""FastAPI application over the existing index, retriever, and answer pipeline.

The index and MiniLM model are loaded once when the process starts. Each
question then calls ``answer_question`` with the same defaults as the command
line: hybrid retrieval, top 10, ``gpt-5.4-mini``, and reasoning effort ``none``.
The OpenAI key stays in the environment or in the project ``.env`` file.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from ..answer import Citation, GroundedAnswer, answer_question
from ..ask import _load_local_env
from ..config import DEFAULT_EXTRACTED_DIR, DEFAULT_INDEX_DIR
from ..indexing.index import IndexUnavailable
from ..ingestion.service import IngestionError, IngestionService

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
Loader = Callable[[], tuple[Any, Any]]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    document: str | None = None


def document_entries(retriever) -> list[dict[str, str]]:
    """One entry per indexed document. ``id`` is stable; ``path`` distinguishes duplicate names."""
    documents = getattr(getattr(retriever, "index", None), "documents", {}) or {}
    entries = []
    for doc_id, info in documents.items():
        name = str(info.get("doc_name") or "").strip()
        path = str(info.get("source_path") or "").strip()
        if not name and not path:
            continue
        entries.append({"id": str(doc_id), "name": name, "path": path})
    entries.sort(key=lambda item: ((item["path"] or item["name"]).lower(), item["id"]))
    return entries


def resolve_document(retriever, selected: str) -> str:
    """Map a document id, exact source path, or unique file name to one document id."""
    documents = getattr(getattr(retriever, "index", None), "documents", {}) or {}
    if selected in documents:
        return selected
    by_path = [doc_id for doc_id, info in documents.items() if str(info.get("source_path") or "") == selected]
    if len(by_path) == 1:
        return by_path[0]
    if len(by_path) > 1:
        raise HTTPException(status_code=409, detail="Several documents share that path. Choose one from the list.")
    by_name = [doc_id for doc_id, info in documents.items() if str(info.get("doc_name") or "") == selected]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise HTTPException(status_code=409, detail="Several documents share that name. Choose one from the list.")
    raise HTTPException(status_code=404, detail=f"No indexed document named '{selected}'.")


def serialize_citation(retriever, citation: Citation) -> dict[str, Any]:
    """Citation text from the answer pipeline, plus location fields from its index unit."""
    unit = None
    primary = getattr(retriever, "primary", {}) or {}
    units = getattr(retriever, "units", []) or []
    index = primary.get(citation.element_id)
    if isinstance(index, int) and 0 <= index < len(units):
        unit = units[index]
    section_path = list(getattr(unit, "section_path", None) or [])
    return {
        "element_id": citation.element_id,
        "citation": citation.citation,
        "kind": citation.kind,
        "document": (getattr(unit, "doc_name", None) or None),
        "page": getattr(unit, "page", None),
        "page_label": getattr(unit, "page_label", None),
        "section": section_path[-1] if section_path else None,
        "label": getattr(unit, "label", None),
    }


def serialize_answer(retriever, result: GroundedAnswer) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "sufficient": result.sufficient,
        "citations": [serialize_citation(retriever, citation) for citation in result.citations],
    }


def load_backend(index_dir: str | Path = DEFAULT_INDEX_DIR) -> tuple[Any, Any]:
    """Load ``.env``, the on-disk index, MiniLM, and one OpenAI client.

    The embedding model is loaded here so later questions do not load it again.
    ``max_retries=0`` keeps a failed request from being sent again automatically.
    """
    _load_local_env()
    from openai import OpenAI

    from ..indexing.embeddings import SentenceTransformerEmbedder
    from ..indexing.index import IndexStore
    from ..retrieval.retrieve import HybridRetriever

    index = IndexStore(index_dir).load()
    embedder = SentenceTransformerEmbedder(allow_download=False)
    embedder.encode(["warm-up"])
    return HybridRetriever(index, embedder), OpenAI(max_retries=0)


def create_app(index_dir: str | Path | None = None, loader: Loader | None = None,
               data_dir: str | Path | None = None, extracted_dir: str | Path | None = None,
               ingestion: IngestionService | None = None) -> FastAPI:
    """Build the application. ``loader`` replaces the real index and client in tests."""
    chosen_index = Path(DEFAULT_INDEX_DIR if index_dir is None else index_dir)
    chosen_data = Path(data_dir) if data_dir is not None else ROOT / "data"
    chosen_extracted = Path(DEFAULT_EXTRACTED_DIR if extracted_dir is None else extracted_dir)
    service = ingestion or IngestionService(chosen_data, chosen_index, chosen_extracted)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = None
        app.state.client = None
        app.state.index_error = None
        app.state.ask_lock = threading.Lock()
        app.state.ingestion = service

        def publish(retriever) -> None:
            with app.state.ask_lock:
                app.state.retriever = retriever
                app.state.index_error = None

        service.on_ready = publish
        try:
            retriever, client = load_backend(chosen_index) if loader is None else loader()
        except IndexUnavailable as exc:
            app.state.index_error = str(exc)
            service.note_missing(str(exc))
        else:
            app.state.retriever = retriever
            app.state.client = client
            service.note_active(retriever)
        yield

    app = FastAPI(title="EventWise AI", lifespan=lifespan)

    def require_index():
        if app.state.retriever is None:
            detail = app.state.index_error or "The search index is not available."
            raise HTTPException(status_code=503, detail=detail)
        return app.state.retriever

    def ensure_questionable():
        current = service.status()
        if current["state"] == "processing":
            raise HTTPException(
                status_code=503,
                detail="The knowledge base is updating. Try again when it is Ready.",
            )
        if not current["previous_available"]:
            raise HTTPException(status_code=503, detail=current["error"] or "The search index is not available.")

    @app.get("/api/ingestion/status")
    def ingestion_status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/documents")
    def list_documents() -> dict[str, list[dict[str, str]]]:
        return {"documents": document_entries(require_index())}

    @app.post("/api/documents")
    def upload_documents(files: list[UploadFile] = File(...)) -> JSONResponse:
        payload = [(item.filename or "", item.file) for item in files]
        try:
            saved = service.submit(payload)
        except IngestionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return JSONResponse(status_code=202, content={"saved": saved, "status": service.status()})

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Enter a question.")
        ensure_questionable()
        retriever = require_index()
        document = (body.document or "").strip()
        if document:
            document = resolve_document(retriever, document)
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not set. Add it to .env in the project directory and restart.",
            )
        try:
            with app.state.ask_lock:
                result = answer_question(
                    retriever,
                    question,
                    app.state.client,
                    documents=[document] if document else None,
                )
        except Exception:
            log.exception("Answer request failed")
            raise HTTPException(status_code=502, detail="The model request failed. Try again in a moment.")
        return serialize_answer(retriever, result)

    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        @app.get("/")
        def frontend() -> FileResponse:
            return FileResponse(index_file)

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    return app


app = create_app()
