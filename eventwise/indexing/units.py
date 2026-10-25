"""Turn extracted elements into searchable index units.

One element yields one or more units; every unit keeps the element ID and the
element's source metadata, so a hit can always be traced back to (and
expanded into) the original structured element:

- heading/text: the element text, split into overlapping windows only when it is
  longer than the embedding model can read;
- table: one unit for the whole table (caption, header, all rows) plus, for large
  tables, row-group units that repeat the caption and column header. The
  structured ``TableData`` stays untouched in ``document.json``;
- figure: caption/label, text found inside the figure, native chart data, and
  short context from text that cites the figure or sits next to it.

Every unit also carries a context prefix (document title and section path).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..extraction.captions import find_label_mentions, label_key, mention_offsets
from ..config import IndexConfig

UNITS_VERSION = "units-v3"  # bump when unit construction changes, to invalidate cached embeddings

_LEADER_LINE = re.compile(r"(?:\.\s?){4,}\s*\S{0,6}\s*$|\u2026{2,}\s*\S{0,6}\s*$")
_BAD_TITLE = re.compile(r"^(untitled|document\d*|microsoft word\b.*|.*\.(docx?|pdf))$", re.IGNORECASE)


@dataclass
class IndexUnit:
    unit_id: str
    element_id: str
    doc_id: str
    doc_name: str
    kind: str
    order: int
    content: str  # the searchable body of this unit (what a reader would quote)
    context: str  # document title + section path, prepended for search
    part: str | None = None  # e.g. "chunk 2/3", "rows 8-13"
    rows: list[int] | None = None  # [first, last] grid rows (0-based, inclusive) for row groups
    page: int | None = None
    page_label: str | None = None
    section_path: list[str] = field(default_factory=list)
    label: str | None = None
    caption: str | None = None
    caption_confidence: str | None = None
    bbox: list[float] | None = None
    location: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None  # relative to the extracted-documents directory
    refs: list[str] = field(default_factory=list)  # element IDs this element cites by label
    cited_by: list[str] = field(default_factory=list)  # element IDs whose text cites this element
    continues: list[str] = field(default_factory=list)  # possible cross-page table partners
    context_sources: list[str] = field(default_factory=list)  # element IDs borrowed as context
    low_text: bool = False  # little or no searchable text of its own
    navigation: bool = False  # table-of-contents style line ("3.1 Requests ....... 19")

    @property
    def search_text(self) -> str:
        return f"{self.context}\n{self.content}" if self.context else self.content

    @property
    def is_primary(self) -> bool:
        """The unit that represents the element as a whole (first chunk / whole table)."""
        return self.rows is None and (self.part is None or self.part.startswith("chunk 1/"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexUnit":
        return cls(**data)


def document_title(doc: dict[str, Any]) -> str:
    stem = Path(doc["doc_name"]).stem
    readable_stem = re.sub(r"[_]+", " ", stem).strip()
    title = (doc.get("metadata") or {}).get("title") or ""
    title = " ".join(title.split())
    if len(title) < 3 or _BAD_TITLE.match(title):
        return readable_stem
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    if squash(title) in squash(readable_stem):
        return title
    return f"{title} ({readable_stem})"


def _words(text: str) -> list[str]:
    return text.split()


def _truncate(text: str, max_words: int) -> str:
    words = _words(text)
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " …"


def _around(text: str, offset: int, max_words: int) -> str:
    """About ``max_words`` words that end shortly after ``offset`` (the words leading up to a citation)."""
    before = text[:offset].split()
    after = text[offset:].split()
    tail = after[: max(4, max_words // 6)]
    head = before[-(max_words - len(tail)):] if max_words > len(tail) else []
    prefix = "… " if len(head) < len(before) else ""
    suffix = " …" if len(tail) < len(after) else ""
    return prefix + " ".join(head + tail) + suffix


def _windows(text: str, size: int, overlap: int) -> list[str]:
    words = _words(text)
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, dict):
        return (cell.get("text") or "").strip()
    return str(cell).strip()


def _table_rows(table: dict[str, Any]) -> list[str]:
    grid = table.get("normalised_cells") or table.get("raw_cells") or []
    return [" | ".join(t for t in map(_cell_text, row) if t) for row in grid]


def _context_prefix(title: str, section_path: list[str]) -> str:
    lines = [f"Document: {title}"]
    if section_path:
        lines.append("Section: " + " > ".join(section_path))
    return "\n".join(lines)


class _DocUnits:
    def __init__(self, doc: dict[str, Any], config: IndexConfig):
        self.doc = doc
        self.config = config
        self.elements: list[dict[str, Any]] = sorted(doc.get("elements") or [], key=lambda e: e["order"])
        self.by_id = {e["element_id"]: e for e in self.elements}
        self.title = document_title(doc)
        self._link_labels()
        self.continuation = self._continuations()

    def _link_labels(self) -> None:
        by_label: dict[str, list[str]] = {}
        for e in self.elements:
            if e["kind"] in ("table", "figure") and e.get("label"):
                key = label_key(e["label"])
                if key:
                    by_label.setdefault(key, []).append(e["element_id"])
        self.refs: dict[str, list[str]] = {}
        self.cited_by: dict[str, list[str]] = {}
        for e in self.elements:
            own = label_key(e["label"]) if e.get("label") else None
            body = e["text"]
            if e.get("caption") and body.startswith(e["caption"]):
                body = body[len(e["caption"]):]  # an element's own caption is not a citation
            targets = []
            for key in sorted(find_label_mentions(body)):
                if key == own:
                    continue
                targets.extend(t for t in by_label.get(key, []) if t != e["element_id"])
            if targets:
                self.refs[e["element_id"]] = targets
                for t in targets:
                    self.cited_by.setdefault(t, []).append(e["element_id"])

    def _continuations(self) -> dict[str, list[str]]:
        links: dict[str, list[str]] = {}
        for issue in self.doc.get("issues") or []:
            if issue.get("code") != "possible_table_continuation":
                continue
            ids = (issue.get("location") or {}).get("element_ids") or []
            for a in ids:
                links.setdefault(a, []).extend(b for b in ids if b != a)
        return links

    def _neighbour_text(self, element: dict[str, Any]) -> list[tuple[str, str]]:
        """(element_id, snippet) of text that cites the element, else the nearest text in its section."""
        cfg = self.config
        found: list[tuple[str, str]] = []
        key = label_key(element["label"]) if element.get("label") else None
        for eid in self.cited_by.get(element["element_id"], []):
            other = self.by_id[eid]
            if other["kind"] == "text":
                offsets = mention_offsets(other["text"], key) if key else []
                snippet = (_around(other["text"], offsets[0], cfg.context_words) if offsets
                           else _truncate(other["text"], cfg.context_words))
                found.append((eid, snippet))
        if not found:
            index = self.elements.index(element)
            for step in (-1, 1, -2, 2, -3, 3):
                j = index + step
                if 0 <= j < len(self.elements):
                    other = self.elements[j]
                    if other["kind"] == "text" and other.get("section_path") == element.get("section_path"):
                        found.append((other["element_id"], _truncate(other["text"], cfg.context_words)))
                        break
        return found[: cfg.max_context_snippets]

    def _base(self, e: dict[str, Any]) -> dict[str, Any]:
        figure = e.get("figure") or {}
        image = figure.get("image_path")
        match = e.get("caption_match") or {}
        return dict(
            element_id=e["element_id"], doc_id=self.doc["doc_id"], doc_name=self.doc["doc_name"], kind=e["kind"],
            order=e["order"], page=e.get("page"), page_label=e.get("page_label"),
            section_path=list(e.get("section_path") or []), label=e.get("label"), caption=e.get("caption"),
            caption_confidence=match.get("confidence") if match.get("status") == "matched" else None,
            bbox=list(e["bbox"]) if e.get("bbox") else None, location=dict(e.get("location") or {}),
            image_path=f"{self.doc['doc_id']}/{image}" if image else None,
            refs=self.refs.get(e["element_id"], []), cited_by=self.cited_by.get(e["element_id"], []),
            continues=self.continuation.get(e["element_id"], []),
            context=_context_prefix(self.title, list(e.get("section_path") or [])),
        )

    def build(self) -> list[IndexUnit]:
        units: list[IndexUnit] = []
        for e in self.elements:
            kind = e["kind"]
            if kind == "table" and e.get("table"):
                units.extend(self._table(e))
            elif kind == "figure":
                units.append(self._figure(e))
            else:
                units.extend(self._text(e))
        return units

    def _text(self, e: dict[str, Any]) -> list[IndexUnit]:
        text = e["text"].strip()
        base = self._base(e)
        if not text:
            return [IndexUnit(unit_id=e["element_id"], content="", low_text=True, **base)]
        lines = [ln for ln in (e.get("raw_text") or text).splitlines() if ln.strip()]
        navigation = bool(lines) and sum(bool(_LEADER_LINE.search(ln)) for ln in lines) * 2 > len(lines)
        pieces = _windows(text, self.config.max_unit_words, self.config.chunk_overlap_words)
        if len(pieces) == 1:
            return [IndexUnit(unit_id=e["element_id"], content=text, low_text=len(_words(text)) < 3,
                              navigation=navigation, **base)]
        n = len(pieces)
        return [IndexUnit(unit_id=f"{e['element_id']}#c{i}", content=p, part=f"chunk {i + 1}/{n}",
                          navigation=navigation, **base) for i, p in enumerate(pieces)]

    def _table(self, e: dict[str, Any]) -> list[IndexUnit]:
        cfg = self.config
        table = e["table"]
        base = self._base(e)
        rows = _table_rows(table)
        n_header = min(max(int(table.get("header_rows") or 0), 0), len(rows))
        header = " / ".join(r for r in rows[:n_header] if r)
        body = [(i, r) for i, r in enumerate(rows) if i >= n_header and r]
        title = e.get("caption") or e.get("label") or ""

        borrowed: list[tuple[str, str]] = [] if title else self._neighbour_text(e)
        intro = [title] if title else []
        if borrowed:
            intro.append("Context: " + " … ".join(s for _, s in borrowed))
        base["context_sources"] = [eid for eid, _ in borrowed]

        whole = "\n".join(intro + ([header] if header else []) + [r for _, r in body])
        units = [IndexUnit(unit_id=e["element_id"], content=whole, low_text=not body and not title, **base)]

        n_words = len(_words(whole))
        if len(body) <= cfg.table_rows_threshold and n_words <= cfg.table_words_threshold:
            return units
        head = intro + ([f"Columns: {header}"] if header else [])
        group: list[tuple[int, str]] = []

        def flush() -> None:
            if not group:
                return
            first, last = group[0][0], group[-1][0]
            content = "\n".join(head + [r for _, r in group])
            units.append(IndexUnit(unit_id=f"{e['element_id']}#r{first}-{last}", content=content,
                                   part=f"rows {first + 1}-{last + 1} of {len(rows)}", rows=[first, last], **base))
            group.clear()

        words = 0
        for i, row in body:
            row_words = len(_words(row))
            if group and (len(group) >= cfg.table_group_rows or words + row_words > cfg.table_group_max_words):
                flush()
                words = 0
            group.append((i, row))
            words += row_words
        flush()
        return units

    def _figure(self, e: dict[str, Any]) -> IndexUnit:
        cfg = self.config
        figure = e.get("figure") or {}
        base = self._base(e)
        parts: list[str] = []
        title = e.get("caption") or e.get("label")
        if title:
            parts.append(title)
        labels = (figure.get("text_labels") or "").strip()
        if labels:
            parts.append("Text in figure: " + _truncate(labels, cfg.figure_label_words))
        elif e["text"].strip() and e["text"].strip() != title:
            parts.append(_truncate(e["text"].strip(), cfg.figure_label_words))
        for series in figure.get("chart_data") or []:
            name = series.get("name") or "series"
            points = ", ".join(f"{c}: {v}" for c, v in zip(series.get("categories") or [], series.get("values") or []))
            parts.append(f"Chart data {name}: {points}")
        low_text = not parts
        borrowed = self._neighbour_text(e)
        if borrowed:
            parts.append("Context: " + " … ".join(s for _, s in borrowed))
        base["context_sources"] = [eid for eid, _ in borrowed]
        if not parts:
            parts.append(f"Figure without caption or text ({figure.get('source_type') or 'image'})")
        return IndexUnit(unit_id=e["element_id"], content="\n".join(parts), low_text=low_text, **base)


def build_units(doc: dict[str, Any], config: IndexConfig | None = None) -> list[IndexUnit]:
    """Index units for one extracted document (a ``document.json`` dict)."""
    if doc.get("status") == "failed":
        return []
    return _DocUnits(doc, config or IndexConfig()).build()


def units_fingerprint(units: list[IndexUnit], config: IndexConfig) -> str:
    """Changes whenever anything that is embedded changes."""
    h = hashlib.sha256(f"{UNITS_VERSION}|{sorted(config.to_dict().items())}".encode())
    for u in units:
        h.update(u.unit_id.encode())
        h.update(b"\0")
        h.update(u.search_text.encode())
        h.update(b"\1")
    return h.hexdigest()[:20]
