"""Unified, format-independent representation of an extracted document.

Every piece of content (heading, paragraph, table, figure) becomes an
``Element`` that carries its own source location, so later stages can
retrieve and cite text, tables and figures in the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ElementKind = Literal["heading", "text", "table", "figure"]
Severity = Literal["info", "warning", "error"]

BBox = tuple[float, float, float, float]


@dataclass
class Issue:
    """Something the parser could not handle fully. Never silently dropped."""

    code: str
    message: str
    severity: Severity = "warning"
    location: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalisedCell:
    text: str
    value: float | None = None
    unit: str | None = None


@dataclass
class CellMerge:
    row: int
    col: int
    row_span: int
    col_span: int
    inferred: bool = False  # True when the span was guessed from layout, not declared


@dataclass
class TableData:
    n_rows: int
    n_cols: int
    # Text exactly as extracted. None marks a grid position covered by a merged cell.
    raw_cells: list[list[str | None]]
    # Parallel grid with whitespace-normalised text and parsed numbers.
    normalised_cells: list[list[NormalisedCell | None]]
    merges: list[CellMerge] = field(default_factory=list)
    cell_bboxes: list[list[BBox | None]] | None = None  # PDF only
    header_rows: int = 1  # best guess; downstream code should treat it as a hint
    method: str = ""


@dataclass
class FigureData:
    source_type: str  # "raster" | "vector" | "embedded_image" | "native_chart"
    image_path: str | None = None  # relative to the document's output directory
    image_format: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    text_labels: str | None = None  # text found inside the figure region (PDF)
    chart_data: list[dict[str, Any]] | None = None  # native DOCX charts only


@dataclass
class CaptionMatch:
    status: Literal["matched", "unmatched"]
    confidence: Literal["high", "medium", "low"] | None = None
    method: str = ""
    distance: float | None = None
    caption_location: dict[str, Any] = field(default_factory=dict)


@dataclass
class Element:
    element_id: str
    kind: ElementKind
    order: int
    text: str  # normalised text (tables: pipe-separated rows)
    raw_text: str | None = None
    page: int | None = None  # 1-based physical PDF page
    page_label: str | None = None  # printed page number, if detected
    bbox: BBox | None = None
    section_path: list[str] = field(default_factory=list)
    heading_level: int | None = None
    label: str | None = None  # e.g. "Table 3", "Figure 1.1.2", "Tabl 1"
    caption: str | None = None
    caption_match: CaptionMatch | None = None
    location: dict[str, Any] = field(default_factory=dict)  # format-specific source refs
    table: TableData | None = None
    figure: FigureData | None = None


@dataclass
class ExtractedDocument:
    doc_id: str
    doc_name: str
    source_path: str
    file_type: str
    status: Literal["ok", "partial", "failed"] = "ok"
    page_count: int | None = None
    pages_processed: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)
    furniture: list[dict[str, Any]] = field(default_factory=list)  # removed headers/footers
    form_fields: list[dict[str, Any]] = field(default_factory=list)  # checkboxes etc.
    elements: list[Element] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def add_issue(self, code: str, message: str, severity: Severity = "warning", **location: Any) -> None:
        self.issues.append(Issue(code, message, severity, location))

    def by_kind(self, kind: ElementKind) -> list[Element]:
        return [e for e in self.elements if e.kind == kind]

    def stats(self) -> dict[str, Any]:
        tables, figures = self.by_kind("table"), self.by_kind("figure")
        captioned = lambda els: sum(1 for e in els if e.caption)  # noqa: E731
        return {
            "status": self.status,
            "pages": len(self.pages_processed or []) or None,
            "headings": len(self.by_kind("heading")),
            "text": len(self.by_kind("text")),
            "tables": len(tables),
            "tables_captioned": captioned(tables),
            "figures": len(figures),
            "figures_captioned": captioned(figures),
            "unmatched_captions": sum(1 for i in self.issues if i.code == "caption_unmatched"),
            "issues": {s: sum(1 for i in self.issues if i.severity == s) for s in ("info", "warning", "error")},
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
