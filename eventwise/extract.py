"""Extraction entry point: dispatch by file type, never raise, save as JSON.

Command line:
    python -m eventwise.extract PATH [PATH ...] [--out DIR] [--pages 1-20,26]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .models import ExtractedDocument
from .parsers import parse_docx, parse_pdf

SUPPORTED = (".pdf", ".docx")


def content_doc_id(data: bytes) -> str:
    """Document ID = content hash, so a changed file gets a new ID."""
    return hashlib.sha256(data).hexdigest()[:16]


def file_doc_id(path: str | Path) -> str:
    return content_doc_id(Path(path).read_bytes())


def load_document_dict(path: str | Path) -> dict:
    """Read a saved ``document.json`` (raises on missing or corrupt files)."""
    return json.loads(Path(path).read_text())


def extract_document(path: str | Path, out_dir: str | Path | None = None,
                     pages: list[int] | None = None) -> ExtractedDocument:
    """Extract one file. Failures are reported in ``issues`` rather than raised.

    ``out_dir`` receives figure images (under ``<out_dir>/<doc_id>/images``);
    when None, no files are written. ``pages`` (1-based) limits PDF processing.
    """
    path = Path(path)
    doc = ExtractedDocument(doc_id="", doc_name=path.name, source_path=str(path),
                            file_type=path.suffix.lower().lstrip("."))
    try:
        data = path.read_bytes()
    except OSError as exc:
        doc.status = "failed"
        doc.add_issue("unreadable_file", f"Cannot read file: {exc}", severity="error")
        return doc
    doc.doc_id = content_doc_id(data)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        doc.status = "failed"
        hint = " Convert legacy .doc files to .docx first." if suffix == ".doc" else ""
        doc.add_issue("unsupported_file_type", f"Only PDF and DOCX are supported, got '{suffix}'.{hint}",
                      severity="error")
        return doc

    doc_dir = Path(out_dir) / doc.doc_id if out_dir is not None else None
    try:
        if suffix == ".pdf":
            parse_pdf(path, doc, doc_dir, pages)
        else:
            if pages:
                doc.add_issue("pages_ignored", "Page selection applies to PDF only; whole DOCX extracted",
                              severity="info")
            parse_docx(path, doc, doc_dir)
    except Exception as exc:
        doc.status = "failed"
        doc.elements.clear()
        doc.add_issue("parse_failed", f"{type(exc).__name__}: {exc}", severity="error")
        return doc

    if doc.status != "failed" and any(i.severity == "error" for i in doc.issues):
        doc.status = "partial"
    return doc


def save_document(doc: ExtractedDocument, out_dir: str | Path) -> Path:
    target = Path(out_dir) / (doc.doc_id or "failed") / "document.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc.to_dict(), ensure_ascii=False, indent=1))
    return target


def parse_page_spec(spec: str) -> list[int]:
    """'1-3,7' -> [1, 2, 3, 7]"""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        elif part:
            pages.add(int(part))
    return sorted(pages)


def collect_files(paths: list[str | Path]) -> list[Path]:
    """Supported files under the given files/directories (directories searched recursively)."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED and not f.name.startswith(".")))
        else:
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract text, tables and figures from PDF/DOCX files.")
    parser.add_argument("paths", nargs="+", help="Files or directories (searched recursively)")
    parser.add_argument("--out", default=".cache/extracted", help="Output directory (default: %(default)s)")
    parser.add_argument("--pages", help="PDF pages to process, e.g. '1-20,26' (default: all)")
    args = parser.parse_args(argv)

    pages = parse_page_spec(args.pages) if args.pages else None
    files = collect_files(args.paths)
    if not files:
        print("No PDF/DOCX files found.", file=sys.stderr)
        return 1

    failures = 0
    print(f"{'file':60} {'status':8} {'pages':>5} {'head':>4} {'text':>5} {'tab':>4} {'fig':>4} {'uncap':>5} {'warn':>4} {'err':>3} {'sec':>6}")
    for path in files:
        start = time.perf_counter()
        doc = extract_document(path, args.out, pages)
        save_document(doc, args.out)
        s = doc.stats()
        failures += doc.status == "failed"
        print(f"{path.name[:60]:60} {s['status']:8} {s['pages'] or '-':>5} {s['headings']:>4} {s['text']:>5} "
              f"{s['tables']:>4} {s['figures']:>4} {s['unmatched_captions']:>5} {s['issues']['warning']:>4} "
              f"{s['issues']['error']:>3} {time.perf_counter() - start:6.1f}")
    print(f"\nWrote results to {args.out}/<doc_id>/document.json")
    return 1 if failures == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
