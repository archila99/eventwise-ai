"""Persistent, incremental search index (units + embeddings) on local disk.

Layout of ``<index_dir>``::

    manifest.json                 documents, embedder, settings
    docs/<doc_id>/units.json      index units (element IDs + source metadata)
    docs/<doc_id>/vectors.npy     float32 [n_units, dim], L2-normalised, row i = unit i

Documents are keyed by content hash (the Stage 1 ``doc_id``), so:
- unchanged files reuse their stored vectors (no model load at all);
- new files are extracted (if not already cached) and embedded;
- edited files get a new ID: the old entry is dropped and the new one indexed;
- deleted files are dropped from the index.

Command line:
    python -m eventwise.index PATH [PATH ...] [--index DIR] [--extracted DIR] [--prune] [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_EXTRACTED_DIR, DEFAULT_INDEX_DIR, IndexConfig
from .embeddings import Embedder
from .units import UNITS_VERSION, IndexUnit, build_units, units_fingerprint

FORMAT_VERSION = 1


class IndexUnavailable(RuntimeError):
    """The index is missing or unusable; rebuild it with ``python -m eventwise.index``."""


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_json(target: Path, data: Any) -> None:
    _atomic_write_bytes(target, json.dumps(data, ensure_ascii=False, indent=1).encode())


def _atomic_save_npy(target: Path, array: np.ndarray) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, array)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class LoadedIndex:
    units: list[IndexUnit]
    vectors: np.ndarray  # [len(units), dim]
    manifest: dict[str, Any]
    index_dir: Path
    extracted_dir: Path
    problems: list[str] = field(default_factory=list)  # documents skipped because their files are unusable

    @property
    def embedder_name(self) -> str:
        return self.manifest.get("embedder", "")

    @property
    def documents(self) -> dict[str, dict[str, Any]]:
        return self.manifest.get("documents", {})


class IndexStore:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.root = Path(index_dir)
        self.manifest_path = self.root / "manifest.json"

    def doc_dir(self, doc_id: str) -> Path:
        return self.root / "docs" / doc_id

    def read_manifest(self) -> dict[str, Any]:
        """Raises IndexUnavailable when missing or unreadable."""
        if not self.manifest_path.exists():
            raise IndexUnavailable(f"No index at {self.root}. Build it with: python -m eventwise.index data/")
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise IndexUnavailable(f"Index manifest {self.manifest_path} is corrupt ({exc}). Rebuild with --rebuild.") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_VERSION or "documents" not in manifest:
            raise IndexUnavailable(f"Index manifest {self.manifest_path} has an unknown format. Rebuild with --rebuild.")
        return manifest

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        _atomic_write_json(self.manifest_path, manifest)

    def save_doc(self, doc_id: str, units: list[IndexUnit], vectors: np.ndarray) -> None:
        if len(units) != len(vectors):
            raise ValueError("units and vectors differ in length")
        target = self.doc_dir(doc_id)
        _atomic_save_npy(target / "vectors.npy", np.ascontiguousarray(vectors, dtype=np.float32))
        _atomic_write_json(target / "units.json", [u.to_dict() for u in units])

    def load_doc(self, doc_id: str, n_units: int, dim: int) -> tuple[list[IndexUnit], np.ndarray]:
        """Raises ValueError/OSError when the stored files are missing, corrupt or inconsistent."""
        target = self.doc_dir(doc_id)
        units = [IndexUnit.from_dict(d) for d in json.loads((target / "units.json").read_text())]
        vectors = np.load(target / "vectors.npy", allow_pickle=False)
        if vectors.dtype != np.float32 or vectors.shape != (n_units, dim) or len(units) != n_units:
            raise ValueError(f"shape {vectors.shape}/{len(units)} units, expected ({n_units}, {dim})")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite values in vectors")
        return units, vectors

    def files_ok(self, doc_id: str, n_units: int, dim: int) -> bool:
        try:
            self.load_doc(doc_id, n_units, dim)
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def drop_doc(self, doc_id: str) -> None:
        shutil.rmtree(self.doc_dir(doc_id), ignore_errors=True)

    def load(self) -> LoadedIndex:
        manifest = self.read_manifest()
        dim = int(manifest.get("dim") or 0)
        units: list[IndexUnit] = []
        blocks: list[np.ndarray] = []
        problems: list[str] = []
        for doc_id, info in manifest["documents"].items():
            try:
                doc_units, vectors = self.load_doc(doc_id, int(info["n_units"]), dim)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{info.get('doc_name', doc_id)}: index files unusable ({exc}); re-run indexing")
                continue
            units.extend(doc_units)
            blocks.append(vectors)
        vectors = np.vstack(blocks) if blocks else np.zeros((0, dim), dtype=np.float32)
        return LoadedIndex(units=units, vectors=vectors, manifest=manifest, index_dir=self.root,
                           extracted_dir=Path(manifest.get("extracted_dir") or DEFAULT_EXTRACTED_DIR),
                           problems=problems)


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    reembedded: list[str] = field(default_factory=list)  # same document, units or embedder changed
    changed: list[str] = field(default_factory=list)  # source file content changed
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extracted: int = 0
    units_total: int = 0
    units_embedded: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        t = self.timings
        lines = [
            f"documents: {len(self.added)} added, {len(self.changed)} changed, {len(self.reembedded)} re-embedded, "
            f"{len(self.reused)} reused from cache, {len(self.removed)} removed, {len(self.failed)} failed",
            f"units: {self.units_total} in index, {self.units_embedded} embedded this run; "
            f"{self.extracted} document(s) extracted this run",
            "time: " + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()),
        ]
        lines += [f"  removed: {n}" for n in self.removed]
        lines += [f"  changed: {n}" for n in self.changed]
        lines += [f"  FAILED: {n}: {why}" for n, why in self.failed]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _norm(path: str | Path) -> Path:
    """One spelling per file: relative to the working directory when inside it, else absolute."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return resolved


def _stat_key(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _load_or_extract(path: Path, doc_id: str, extracted_dir: Path, report: SyncReport) -> dict[str, Any] | None:
    from .extract import extract_document, load_document_dict, save_document  # parsers load lazily

    cached = extracted_dir / doc_id / "document.json"
    if cached.exists():
        try:
            doc = load_document_dict(cached)
            if doc.get("doc_id") == doc_id:
                return doc
            report.notes.append(f"{path.name}: cached extraction has a different doc_id; re-extracting")
        except (OSError, ValueError) as exc:
            report.notes.append(f"{path.name}: cached extraction unreadable ({exc}); re-extracting")
    extracted = extract_document(path, extracted_dir)
    report.extracted += 1
    if extracted.status == "failed":
        reasons = "; ".join(i.message for i in extracted.issues if i.severity == "error") or "extraction failed"
        report.failed.append((str(path), reasons))
        return None
    save_document(extracted, extracted_dir)
    return extracted.to_dict()


def sync_index(paths: list[str | Path], embedder: Embedder, index_dir: str | Path = DEFAULT_INDEX_DIR,
               extracted_dir: str | Path = DEFAULT_EXTRACTED_DIR, config: IndexConfig | None = None,
               prune: bool = False, rebuild: bool = False, batch_size: int = 64) -> SyncReport:
    """Bring the index in line with the given files; reuse everything that has not changed.

    ``prune=True`` also removes indexed documents that are not among ``paths``.
    Documents whose source file was deleted or edited are always updated.
    """
    from .extract import collect_files, file_doc_id

    config = config or IndexConfig()
    store = IndexStore(index_dir)
    extracted_dir = Path(extracted_dir)
    report = SyncReport()
    start = time.perf_counter()

    old: dict[str, Any] = {}
    if not rebuild:
        try:
            old = store.read_manifest()
        except IndexUnavailable as exc:
            if store.manifest_path.exists():
                report.notes.append(f"{exc} Starting a fresh index.")
    old_docs: dict[str, dict[str, Any]] = dict(old.get("documents", {}))
    same_embedder = old.get("embedder") == embedder.name and old.get("units_version") == UNITS_VERSION

    # 1. Which files are wanted, and what is their content ID?
    t0 = time.perf_counter()
    files = list(dict.fromkeys(_norm(p) for p in collect_files(paths)))
    for info in old_docs.values():
        info["source_path"] = str(_norm(info["source_path"]))
    by_path = {info["source_path"]: doc_id for doc_id, info in old_docs.items()}
    for doc_id, info in old_docs.items():  # edited or deleted sources, even outside ``paths``
        src = Path(info["source_path"])
        if not src.exists():
            continue
        if src not in files and _stat_key(src) != info.get("stat"):
            files.append(src)
    wanted: dict[str, Path] = {}
    stats: dict[str, list[int]] = {}
    for path in files:
        try:
            key = _stat_key(path)
            known = by_path.get(str(path))
            doc_id = known if known and old_docs[known].get("stat") == key else file_doc_id(path)
        except OSError as exc:
            report.failed.append((str(path), f"cannot read file: {exc}"))
            continue
        if doc_id in wanted:
            report.notes.append(f"{path} has the same content as {wanted[doc_id]}; indexed once")
            continue
        wanted[doc_id] = path
        stats[doc_id] = key
    report.timings["scan"] = time.perf_counter() - t0

    # 2. Drop deleted, edited and (with prune) unwanted documents.
    wanted_paths = {str(p) for p in wanted.values()}
    for doc_id, info in old_docs.items():
        if doc_id in wanted:
            continue
        src = info["source_path"]
        if not Path(src).exists():
            report.removed.append(info["doc_name"])
        elif src in wanted_paths:
            report.changed.append(info["doc_name"])
        elif not prune:
            continue
        else:
            report.removed.append(info["doc_name"])
        store.drop_doc(doc_id)

    # 3. Units for every wanted document; decide reuse vs embed.
    t0 = time.perf_counter()
    documents: dict[str, dict[str, Any]] = {d: i for d, i in old_docs.items()
                                            if d not in wanted and Path(i["source_path"]).exists()
                                            and not prune and i["source_path"] not in wanted_paths}
    pending: list[tuple[str, list[IndexUnit], dict[str, Any]]] = []
    t_extract = 0.0
    for doc_id, path in wanted.items():
        te = time.perf_counter()
        doc = _load_or_extract(path, doc_id, extracted_dir, report)
        t_extract += time.perf_counter() - te
        if doc is None:
            store.drop_doc(doc_id)
            continue
        units = build_units(doc, config)
        fingerprint = units_fingerprint(units, config)
        entry = {"doc_name": doc["doc_name"], "source_path": str(path), "stat": stats[doc_id],
                 "status": doc.get("status"), "fingerprint": fingerprint, "n_units": len(units)}
        prev = old_docs.get(doc_id)
        if (prev and same_embedder and prev.get("fingerprint") == fingerprint
                and store.files_ok(doc_id, len(units), int(old.get("dim") or 0))):
            entry["indexed_at"] = prev.get("indexed_at")
            documents[doc_id] = entry
            report.reused.append(doc["doc_name"])
            continue
        if prev:
            report.reembedded.append(doc["doc_name"])
        elif doc["doc_name"] not in report.changed:
            report.added.append(doc["doc_name"])
        pending.append((doc_id, units, entry))
    report.timings["extract"] = t_extract
    report.timings["units"] = time.perf_counter() - t0 - t_extract

    # 4. Embed everything new in batches, then save per document.
    t0 = time.perf_counter()
    texts = [u.search_text for _, units, _ in pending for u in units]
    vectors = embedder.encode(texts, batch_size=batch_size) if texts else None
    report.timings["embed"] = time.perf_counter() - t0
    offset = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for doc_id, units, entry in pending:
        block = vectors[offset:offset + len(units)] if vectors is not None else np.zeros((0, embedder.dim), np.float32)
        offset += len(units)
        store.save_doc(doc_id, units, block)
        entry["indexed_at"] = now
        documents[doc_id] = entry
    report.units_embedded = len(texts)

    dim = int(vectors.shape[1]) if vectors is not None and vectors.size else int(old.get("dim") or embedder.dim)
    if not same_embedder and old_docs:
        stale = [d for d in documents if d not in wanted]
        for doc_id in stale:  # vectors from another model cannot be mixed in
            report.notes.append(f"{documents[doc_id]['doc_name']}: dropped (embedded with a different model)")
            store.drop_doc(doc_id)
            documents.pop(doc_id)
    for doc_id in list(old_docs):
        if doc_id not in documents:
            store.drop_doc(doc_id)

    store.write_manifest({
        "format": FORMAT_VERSION, "units_version": UNITS_VERSION, "embedder": embedder.name, "dim": dim,
        "extracted_dir": str(extracted_dir), "index_config": config.to_dict(), "updated_at": now,
        "documents": documents,
    })
    report.units_total = sum(int(d["n_units"]) for d in documents.values())
    report.timings["total"] = time.perf_counter() - start
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or update the local search index.")
    parser.add_argument("paths", nargs="+", help="PDF/DOCX files or directories (searched recursively)")
    parser.add_argument("--index", default=DEFAULT_INDEX_DIR, help="Index directory (default: %(default)s)")
    parser.add_argument("--extracted", default=DEFAULT_EXTRACTED_DIR, help="Extraction cache (default: %(default)s)")
    parser.add_argument("--prune", action="store_true", help="Remove indexed documents not among PATHS")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached embeddings and rebuild everything")
    args = parser.parse_args(argv)

    from .embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    report = sync_index(args.paths, embedder, args.index, args.extracted, prune=args.prune, rebuild=args.rebuild)
    print(report.summary())
    if embedder.load_seconds is not None:
        print(f"model load: {embedder.load_seconds:.1f}s (included in embed time)")
    else:
        print("model not loaded (all embeddings reused from cache)")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
