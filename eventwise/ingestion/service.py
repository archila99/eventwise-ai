"""Save uploads under ``data/`` and publish an index without touching the live one first.

One job runs at a time. Extraction and embedding use the existing ``sync_index``
pipeline, writing into a copy of the index. The live index is replaced only
after that copy loads and still contains the previous documents.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from ..indexing.index import IndexStore, IndexUnavailable, sync_index
from ..retrieval.retrieve import HybridRetriever

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK"
SUPPORTED = {".pdf", ".docx"}


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class IngestionBusy(IngestionError):
    def __init__(self) -> None:
        super().__init__("An update is already running.", 409)


@dataclass
class UploadLimits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_total_bytes: int = 200 * 1024 * 1024


class IngestionService:
    def __init__(self, data_dir: str | Path, index_dir: str | Path, extracted_dir: str | Path,
                 limits: UploadLimits | None = None):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.extracted_dir = Path(extracted_dir)
        self.limits = limits or UploadLimits()
        self.embedder = None
        self.on_ready: Callable[[Any], None] | None = None
        self._lock = threading.Lock()
        self._state = "not_ready"
        self._stage = "not_ready"
        self._documents = 0
        self._units = 0
        self._files_processing = 0
        self._error: str | None = None
        self._previous_available = False
        self._thread: threading.Thread | None = None
        self._accept_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "documents": self._documents,
                "units": self._units,
                "files_processing": self._files_processing,
                "stage": self._stage,
                "error": self._error,
                "previous_available": self._previous_available,
            }

    def note_active(self, retriever) -> None:
        documents = getattr(getattr(retriever, "index", None), "documents", {}) or {}
        units = getattr(retriever, "units", []) or []
        with self._lock:
            self._documents = len(documents)
            self._units = len(units)
            self._previous_available = True
            self.embedder = getattr(retriever, "embedder", self.embedder)
            if self._state != "processing":
                self._state = "ready"
                self._stage = "ready"
                self._error = None

    def note_missing(self, message: str) -> None:
        with self._lock:
            if self._state == "processing":
                return
            self._state = "not_ready"
            self._stage = "not_ready"
            self._error = message
            self._previous_available = False
            self._documents = 0
            self._units = 0

    def submit(self, files: list[tuple[str, BinaryIO]]) -> list[str]:
        """Save files, then start the single background job. A second upload waits, then is refused."""
        with self._accept_lock:
            saved = self.save_uploads(files)
            try:
                self.start(saved)
            except Exception:
                self._rollback([self.data_dir / path for path in saved])
                raise
            return saved

    def save_uploads(self, files: list[tuple[str, BinaryIO]]) -> list[str]:
        """Validate and store files under the data directory. Returns paths relative to it."""
        if not files:
            raise IngestionError("Choose at least one PDF or DOCX file.")
        with self._lock:
            if self._state == "processing":
                raise IngestionBusy()
        planned: list[tuple[Path, str]] = []
        seen: set[Path] = set()
        for raw_name, _handle in files:
            dest = self._destination(raw_name or "")
            if dest in seen:
                raise IngestionError(f"'{self._relative(dest)}' is listed more than once.", 409)
            seen.add(dest)
            if dest.exists():
                raise IngestionError(f"'{self._relative(dest)}' is already in the knowledge base.", 409)
            planned.append((dest, raw_name))
        written: list[Path] = []
        total = 0
        try:
            for dest, _name in planned:
                handle = files[len(written)][1]
                total += self._write_limited(handle, dest, total)
                self._check_signature(dest)
                written.append(dest)
        except Exception:
            self._rollback(written)
            raise
        return [self._relative(path) for path in written]

    def start(self, relative_paths: list[str]) -> None:
        required = [self.data_dir / path for path in relative_paths]
        with self._lock:
            if self._state == "processing":
                raise IngestionBusy()
            self._state = "processing"
            self._stage = "queued"
            self._files_processing = len(required)
            self._error = None
            self._thread = threading.Thread(
                target=self._worker, args=(required,), name="eventwise-ingestion", daemon=True,
            )
            self._thread.start()

    def _worker(self, required: list[Path]) -> None:
        staging: Path | None = None
        published = False
        try:
            embedder = self._embedder_for_job()
            self._set(stage="preparing")
            old_manifest = self._read_manifest()
            staging = self._staging_dir()
            if staging.exists():
                shutil.rmtree(staging)
            if self.index_dir.exists():
                shutil.copytree(self.index_dir, staging)
            else:
                staging.mkdir(parents=True)
            self._set(stage="indexing")
            report = sync_index([self.data_dir], embedder, staging, self.extracted_dir, prune=False)
            if report.failed:
                details = "; ".join(f"{self._display_failure(name)}: {why}" for name, why in report.failed)
                raise RuntimeError(details or "Indexing failed.")
            self._set(stage="validating")
            loaded = IndexStore(staging).load()
            self._check_coverage(old_manifest, loaded, required)
            self._set(stage="publishing")
            self._publish(staging)
            published = True
            staging = None
            active = IndexStore(self.index_dir).load()
            if active.problems:
                raise RuntimeError("The updated index could not be read. " + "; ".join(active.problems))
            retriever = HybridRetriever(active, embedder)
            if self.on_ready is not None:
                self.on_ready(retriever)
            with self._lock:
                self._state = "ready"
                self._stage = "ready"
                self._documents = len(retriever.index.documents)
                self._units = len(retriever.units)
                self._files_processing = 0
                self._error = None
                self._previous_available = True
        except Exception as exc:
            log.exception("Ingestion failed")
            if published:
                self._restore_previous()
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            self._rollback(required)
            with self._lock:
                self._state = "failed"
                self._stage = "failed"
                self._files_processing = 0
                self._error = str(exc) or "Indexing failed."

    def _embedder_for_job(self):
        if self.embedder is not None:
            return self.embedder
        from ..indexing.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(allow_download=False)
        embedder.encode(["warm-up"])
        self.embedder = embedder
        return embedder

    def _set(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def _read_manifest(self) -> dict[str, Any]:
        try:
            return IndexStore(self.index_dir).read_manifest()
        except IndexUnavailable:
            return {}

    def _staging_dir(self) -> Path:
        return self.index_dir.with_name(self.index_dir.name + "-staging")

    def _backup_dir(self) -> Path:
        return self.index_dir.with_name(self.index_dir.name + "-previous")

    def _publish(self, staging: Path) -> None:
        backup = self._backup_dir()
        if backup.exists():
            shutil.rmtree(backup)
        if self.index_dir.exists():
            self.index_dir.rename(backup)
        try:
            staging.rename(self.index_dir)
        except Exception:
            if not self.index_dir.exists() and backup.exists():
                backup.rename(self.index_dir)
            raise

    def _restore_previous(self) -> None:
        backup = self._backup_dir()
        if not backup.exists():
            return
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir, ignore_errors=True)
        if not self.index_dir.exists():
            backup.rename(self.index_dir)

    def _check_coverage(self, old_manifest: dict[str, Any], loaded, required: list[Path]) -> None:
        if loaded.problems:
            raise RuntimeError("The updated index skipped documents. " + "; ".join(loaded.problems))
        sources = _source_paths(loaded.documents)
        ids = set(loaded.documents)
        for doc_id, info in (old_manifest.get("documents") or {}).items():
            raw = info.get("source_path")
            if not raw or not Path(raw).exists():
                continue
            if doc_id in ids or Path(raw).resolve() in sources:
                continue
            name = info.get("doc_name") or doc_id
            raise RuntimeError(f"The updated index is missing {name}.")
        from ..extraction.extract import file_doc_id

        for path in required:
            if not path.exists():
                raise RuntimeError(f"The updated index is missing {self._relative(path)}.")
            if path.resolve() in sources or file_doc_id(path) in ids:
                continue
            raise RuntimeError(f"The updated index is missing {self._relative(path)}.")
        old_units = sum(int(info.get("n_units") or 0) for info in (old_manifest.get("documents") or {}).values())
        if old_units and not loaded.units:
            raise RuntimeError("The updated index is empty.")

    def _destination(self, raw_name: str) -> Path:
        name = raw_name.replace("\\", "/").strip()
        if not name or "\x00" in name or name.startswith("/") or name.startswith("~"):
            raise IngestionError("That file path is not allowed.")
        path = Path(name)
        if path.is_absolute():
            raise IngestionError("That file path is not allowed.")
        parts = path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise IngestionError("That file path is not allowed.")
        if path.suffix.lower() not in SUPPORTED:
            raise IngestionError(f"Only PDF and DOCX files can be added. Unsupported: {path.name}")
        root = self.data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dest = (self.data_dir / path).resolve()
        if dest != root and root not in dest.parents:
            raise IngestionError("That file path is not allowed.")
        return dest

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_dir.resolve()).as_posix()

    def _display_failure(self, name: str) -> str:
        try:
            return self._relative(Path(name))
        except (ValueError, OSError):
            return Path(name).name

    def _write_limited(self, handle: BinaryIO, dest: Path, already: int) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.limits.max_file_bytes:
                        raise IngestionError(
                            f"Each file must be at most {self.limits.max_file_bytes // (1024 * 1024)} MB.",
                            413,
                        )
                    if already + size > self.limits.max_total_bytes:
                        raise IngestionError(
                            f"One upload must be at most {self.limits.max_total_bytes // (1024 * 1024)} MB.",
                            413,
                        )
                    out.write(chunk)
        except Exception:
            dest.unlink(missing_ok=True)
            self._remove_empty_parents(dest.parent)
            raise
        if size == 0:
            dest.unlink(missing_ok=True)
            raise IngestionError(f"'{self._relative(dest)}' is empty.")
        return size

    def _check_signature(self, dest: Path) -> None:
        header = dest.open("rb").read(5)
        suffix = dest.suffix.lower()
        label = dest.name
        try:
            label = self._relative(dest)
        except ValueError:
            pass
        if suffix == ".pdf" and not header.startswith(PDF_MAGIC):
            dest.unlink(missing_ok=True)
            self._remove_empty_parents(dest.parent)
            raise IngestionError(f"'{label}' is not a PDF.")
        if suffix == ".docx" and not header.startswith(DOCX_MAGIC):
            dest.unlink(missing_ok=True)
            self._remove_empty_parents(dest.parent)
            raise IngestionError(f"'{label}' is not a DOCX file.")

    def _rollback(self, written: list[Path]) -> None:
        for path in written:
            path.unlink(missing_ok=True)
            self._remove_empty_parents(path.parent)

    def _remove_empty_parents(self, start: Path) -> None:
        root = self.data_dir.resolve()
        current = start.resolve()
        while current != root and root in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _source_paths(documents: dict[str, dict[str, Any]]) -> set[Path]:
    found = set()
    for info in documents.values():
        raw = info.get("source_path")
        if not raw:
            continue
        try:
            found.add(Path(raw).resolve())
        except OSError:
            continue
    return found
