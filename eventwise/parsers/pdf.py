"""PDF extraction with PyMuPDF.

Pass 1 reads text lines and tables for every requested page. Repeated
header/footer lines are then detected across pages. Pass 2 builds each
page's text blocks, figures and captions in reading order. Pass 3 assigns
heading levels and section paths across the whole document.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..captions import Caption, match_by_geometry, parse_caption
from ..models import (
    BBox,
    CaptionMatch,
    CellMerge,
    Element,
    ExtractedDocument,
    FigureData,
    TableData,
)
from ..normalise import clean_text, normalise_cell

MARGIN_FRACTION = 0.08  # top/bottom band searched for headers, footers, page numbers
BACKGROUND_AREA = 0.85  # images/drawings covering more of the page are decoration
MIN_FIGURE_SIDE_PT = 40
MIN_FIGURE_AREA = 0.01  # fraction of page area
MIN_VECTOR_PATHS = 8  # drawing operations needed before a cluster counts as a figure
RENDER_DPI = 150
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES

PAGE_NUMBER = re.compile(r"^(?:page\s+)?(\d{1,4})(?:\s*(?:/|of)\s*\d{1,4})?$", re.IGNORECASE)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
ID_PREFIX = {"heading": "h", "text": "x", "table": "tab", "figure": "fig"}


@dataclass
class _Line:
    text: str
    bbox: BBox
    size: float
    bold: bool


@dataclass
class _Block:
    lines: list[_Line]
    block_no: int

    @property
    def bbox(self) -> BBox:
        return _union([line.bbox for line in self.lines])

    @property
    def raw_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class _Item:
    """A page-level element before document-wide assembly."""

    kind: str
    element_id: str
    bbox: BBox
    raw_text: str = ""
    lines: list[_Line] = field(default_factory=list)
    block_no: int | None = None
    table: TableData | None = None
    figure: FigureData | None = None
    labels: list[str] = field(default_factory=list)
    caption: Caption | None = None
    caption_match: CaptionMatch | None = None
    location: dict = field(default_factory=dict)


@dataclass
class _Page:
    number: int
    rect: pymupdf.Rect
    blocks: list[_Block]
    tables: list[_Item]
    label: str | None = None


def _round(rect) -> BBox:
    return tuple(round(float(v), 2) for v in rect)  # type: ignore[return-value]


def _union(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _centre_inside(box: BBox, region: BBox, pad: float = 0.0) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] - pad <= cx <= region[2] + pad and region[1] - pad <= cy <= region[3] + pad


def _overlap_fraction(a: BBox, b: BBox) -> float:
    """Intersection area as a fraction of the smaller rectangle."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


def _in_margin(box: BBox, rect: pymupdf.Rect) -> bool:
    band = rect.height * MARGIN_FRACTION
    return box[3] <= rect.y0 + band or box[1] >= rect.y1 - band


# ---------------------------------------------------------------- pass 1


def _read_blocks(page: pymupdf.Page) -> list[_Block]:
    blocks = []
    for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
        if block.get("type") != 0:
            continue
        lines = []
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            main = max(spans, key=lambda s: len(s["text"]))
            bold = all(bool(s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            lines.append(_Line(text, _round(line["bbox"]), round(main["size"], 1), bold))
        if lines:
            blocks.append(_Block(lines, block.get("number", len(blocks))))
    return blocks


def _table_item(page: pymupdf.Page, table, index: int, element_id: str) -> _Item:
    raw: list[list[str | None]] = []
    boxes: list[list[BBox | None]] = []
    for row in table.rows:
        raw_row, box_row = [], []
        for cell in row.cells:
            if cell is None:
                raw_row.append(None)
                box_row.append(None)
            else:
                # Re-reading each cell area keeps word spacing that Table.extract() can lose.
                rect = pymupdf.Rect(cell)
                raw_row.append(page.get_text("text", clip=rect).strip())
                box_row.append(_round(rect))
        raw.append(raw_row)
        boxes.append(box_row)
    n_cols = max((len(r) for r in raw), default=0)
    for raw_row, box_row in zip(raw, boxes):
        raw_row.extend([None] * (n_cols - len(raw_row)))
        box_row.extend([None] * (n_cols - len(box_row)))

    # Thick or double rules can create empty columns a couple of points wide.
    slivers = [c for c in range(n_cols)
               if all(not (row[c] or "").strip() for row in raw)
               and all(b[c] is None or b[c][2] - b[c][0] < 3 for b in boxes)]
    if slivers:
        raw = [[v for c, v in enumerate(row) if c not in slivers] for row in raw]
        boxes = [[v for c, v in enumerate(row) if c not in slivers] for row in boxes]
        n_cols -= len(slivers)

    merges = _infer_pdf_merges(raw, boxes)
    normalised = [[normalise_cell(c) for c in row] for row in raw]
    header_external = bool(getattr(table.header, "external", False))
    data = TableData(
        n_rows=len(raw),
        n_cols=n_cols,
        raw_cells=raw,
        normalised_cells=normalised,
        merges=merges,
        cell_bboxes=boxes,
        header_rows=0 if header_external else 1,
        method="pymupdf.find_tables+cell_clip",
    )
    return _Item(
        kind="table",
        element_id=element_id,
        bbox=_round(table.bbox),
        table=data,
        location={"table_index_on_page": index, "header_outside_table": header_external,
                  **({"dropped_empty_sliver_columns": slivers} if slivers else {})},
    )


def _infer_pdf_merges(raw: list[list[str | None]], boxes: list[list[BBox | None]]) -> list[CellMerge]:
    """PyMuPDF reports positions covered by a spanning cell as None. Work out
    which visible cell covers them from the cell geometry."""
    row_mid = []
    for box_row in boxes:
        present = [b for b in box_row if b]
        row_mid.append(sum((b[1] + b[3]) / 2 for b in present) / len(present) if present else None)

    covered: dict[tuple[int, int], tuple[int, int]] = {}
    for r, row in enumerate(raw):
        for c, value in enumerate(row):
            if value is not None:
                continue
            up = r - 1
            while up >= 0 and raw[up][c] is None:
                up -= 1
            above = boxes[up][c] if up >= 0 else None
            if above and row_mid[r] is not None and above[3] > row_mid[r]:
                covered[(r, c)] = (up, c)
                continue
            left = c - 1
            while left >= 0 and row[left] is None:
                left -= 1
            if left >= 0:
                covered[(r, c)] = (r, left)

    spans: dict[tuple[int, int], list[int]] = {}
    for (r, c), (r0, c0) in covered.items():
        span = spans.setdefault((r0, c0), [1, 1])
        span[0] = max(span[0], r - r0 + 1)
        span[1] = max(span[1], c - c0 + 1)
    return [CellMerge(r0, c0, rs, cs, inferred=True) for (r0, c0), (rs, cs) in sorted(spans.items())]


# ------------------------------------------------- headers, footers, page numbers


def _detect_furniture(pages: list[_Page]) -> tuple[set[str], list[dict]]:
    """Find lines repeated in the top/bottom margin across pages."""
    seen: dict[str, set[int]] = defaultdict(set)
    example: dict[str, str] = {}
    for page in pages:
        table_boxes = [t.bbox for t in page.tables]
        numbers: list[str] = []  # a two-page spread carries two printed numbers
        for block in page.blocks:
            for line in block.lines:
                if not _in_margin(line.bbox, page.rect) or any(_centre_inside(line.bbox, t) for t in table_boxes):
                    continue
                text = clean_text(line.text)
                match = PAGE_NUMBER.match(text)
                if match:
                    if match.group(1) not in numbers:
                        numbers.append(match.group(1))
                    continue
                if len(text) > 120 or text.endswith("."):
                    continue  # sentence-like: probably a footnote, which is content
                for key in _furniture_keys(text):
                    seen[key].add(page.number)
                    example.setdefault(key, text)
        if numbers:
            page.label = "-".join(sorted(numbers, key=int))

    n = len(pages)
    repeated = {k for k, p in seen.items() if len(p) >= 3 or (n >= 2 and len(p) >= 2 and len(p) >= 0.5 * n)}
    records = [{"pattern": k, "example": example[k], "pages": sorted(seen[k])} for k in sorted(repeated)]
    return repeated, records


def _furniture_keys(text: str) -> set[str]:
    """The whole line plus its '|'-separated segments, digits masked. Running
    headers often vary by section but share a segment such as the report title."""
    key = re.sub(r"\d+", "#", text.lower())
    segments = {s.strip() for s in key.split("|") if len(s.strip()) >= 8}
    return {key} | segments


def _is_furniture(line: _Line, page: _Page, repeated: set[str]) -> bool:
    if not _in_margin(line.bbox, page.rect):
        return False
    text = clean_text(line.text)
    return bool(PAGE_NUMBER.match(text)) or bool(_furniture_keys(text) & repeated)


# ---------------------------------------------------------------- pass 2


def _find_figures(page: pymupdf.Page, layout: _Page, body_lines: list[_Line], counts: Counter) -> list[_Item]:
    page_area = layout.rect.get_area()
    table_boxes = [t.bbox for t in layout.tables]
    figures: list[_Item] = []

    seen = set()
    for info in page.get_image_info(xrefs=True):
        rect = pymupdf.Rect(info["bbox"]) & layout.rect
        if rect.is_empty:
            continue
        box = _round(rect)
        if box in seen:
            continue
        seen.add(box)
        fraction = rect.get_area() / page_area
        if fraction >= BACKGROUND_AREA:
            counts["background_image_skipped"] += 1
        elif rect.width < MIN_FIGURE_SIDE_PT or rect.height < MIN_FIGURE_SIDE_PT or fraction < MIN_FIGURE_AREA:
            counts["small_image_skipped"] += 1
        elif any(_centre_inside(box, t) for t in table_boxes):
            counts["image_inside_table_skipped"] += 1
        else:
            figures.append(_Item(
                kind="figure", element_id="", bbox=box,
                figure=FigureData("raster", width_px=info.get("width"), height_px=info.get("height")),
                location={"xref": info.get("xref"), "detection": "embedded_image"},
            ))

    blocked = table_boxes + [f.bbox for f in figures]
    drawings = [
        d for d in page.get_drawings()
        if d["rect"].get_area() < BACKGROUND_AREA * page_area
        and not any(_centre_inside(_round(d["rect"]), b, pad=2) for b in blocked)
    ]
    if not drawings:
        return figures
    for cluster in page.cluster_drawings(drawings=drawings):
        box = _round(cluster)
        if cluster.width < MIN_FIGURE_SIDE_PT or cluster.height < MIN_FIGURE_SIDE_PT:
            continue
        if cluster.get_area() / page_area < MIN_FIGURE_AREA:
            continue
        paths = sum(1 for d in drawings if _centre_inside(_round(d["rect"]), box, pad=1))
        if paths < MIN_VECTOR_PATHS:
            continue
        if any(_overlap_fraction(box, other) > 0.5 for other in table_boxes + [f.bbox for f in figures]):
            continue  # table borders or a frame around an image
        inside = [l for l in body_lines if _centre_inside(l.bbox, box)]
        long_lines = sum(1 for l in inside if len(l.text) > 80)
        if long_lines >= 3:  # a shaded text panel, not a chart
            counts["vector_text_panel_skipped"] += 1
            continue
        figures.append(_Item(
            kind="figure", element_id="", bbox=box,
            figure=FigureData("vector"),
            location={"detection": "vector_cluster", "drawing_paths": paths},
        ))
    return figures


def _build_page(page: pymupdf.Page, layout: _Page, repeated: set[str], doc: ExtractedDocument,
                out_dir: Path | None, counts: Counter) -> list[_Item]:
    short = doc.doc_id[:10]
    pno = layout.number

    blocks: list[_Block] = []
    furniture_lines = []
    for block in layout.blocks:
        lines = []
        for line in block.lines:
            (furniture_lines if _is_furniture(line, layout, repeated) else lines).append(line)
        if lines:
            blocks.append(_Block(lines, block.block_no))
    doc.pages[-1]["removed_header_footer"] = [clean_text(l.text) for l in furniture_lines]
    all_lines = [l for b in blocks for l in b.lines]

    figures = _find_figures(page, layout, all_lines, counts)
    objects = layout.tables + figures
    for n, fig in enumerate(figures, 1):
        fig.element_id = f"{short}:p{pno}:fig{n}"

    # Assign lines to tables/figures; anything left is body text.
    body_blocks: list[_Block] = []
    captions: list[tuple[Caption, BBox, _Block]] = []
    for block in blocks:
        keep = []
        for line in block.lines:
            owner = next((o for o in objects if _centre_inside(line.bbox, o.bbox)), None)
            if owner is None:
                keep.append(line)
            elif owner.kind == "figure":
                caption = parse_caption(clean_text(line.text))
                if caption:
                    captions.append((caption, line.bbox, _Block([line], block.block_no)))
                else:
                    owner.labels.append(line.text)
        if not keep:
            continue
        body = _Block(keep, block.block_no)
        caption = parse_caption(clean_text(body.raw_text))
        if caption:
            captions.append((caption, body.bbox, body))
        else:
            body_blocks.append(body)

    assignments = match_by_geometry([(c, b) for c, b, _ in captions], [(o.kind, o.bbox) for o in objects])
    matched = set()
    for a in assignments:
        caption, cbox, _ = captions[a.caption_index]
        obj = objects[a.object_index]
        obj.caption = caption
        obj.caption_match = CaptionMatch("matched", a.confidence, a.method, a.distance,
                                         {"page": pno, "bbox": cbox})
        matched.add(a.caption_index)
        if a.confidence == "low":
            doc.add_issue("caption_low_confidence",
                          f"'{caption.label}' attached to a {obj.kind} although the label suggests a {caption.kind}",
                          element_id=obj.element_id, page=pno)
    for i, (caption, cbox, block) in enumerate(captions):
        if i not in matched:
            doc.add_issue("caption_unmatched", f"No {caption.kind} found near caption '{caption.label}'",
                          page=pno, bbox=cbox)
            body_blocks.append(block)

    for obj in objects:
        if obj.caption_match is None:
            obj.caption_match = CaptionMatch("unmatched")
        if obj.kind == "figure":
            obj.figure.text_labels = clean_text("\n".join(obj.labels)) or None
            if out_dir is not None:
                _save_crop(page, obj, out_dir, doc)

    # Reading order: text blocks keep PDF order; tables/figures are inserted by vertical position.
    body_blocks.sort(key=lambda b: next((i for i, x in enumerate(layout.blocks) if x.block_no == b.block_no), 0))
    pending = sorted(objects, key=lambda o: (o.bbox[1], o.bbox[0]))
    items: list[_Item] = []
    text_n = 0
    for block in body_blocks:
        while pending and pending[0].bbox[1] <= block.bbox[1]:
            items.append(pending.pop(0))
        text_n += 1
        items.append(_Item(kind="text", element_id=f"{short}:p{pno}:x{text_n}", bbox=block.bbox,
                           raw_text=block.raw_text, lines=block.lines, block_no=block.block_no))
    items.extend(pending)

    text_chars = sum(len(l.text) for l in all_lines)
    if text_chars < 20 and any(f.figure.source_type == "raster" for f in figures):
        doc.add_issue("possible_scanned_page", "Page has almost no text layer but contains a large image; OCR was not run",
                      page=pno)
    return items


def _save_crop(page: pymupdf.Page, item: _Item, out_dir: Path, doc: ExtractedDocument) -> None:
    try:
        image_dir = out_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        name = item.element_id.split(":", 1)[1].replace(":", "_") + ".png"
        pix = page.get_pixmap(clip=pymupdf.Rect(item.bbox), dpi=RENDER_DPI)
        pix.save(image_dir / name)
        item.figure.image_path = f"images/{name}"
        item.figure.image_format = "png"
        if item.figure.width_px is None:
            item.figure.width_px, item.figure.height_px = pix.width, pix.height
    except Exception as exc:  # rendering problems should not stop extraction
        doc.add_issue("figure_render_failed", f"Could not render figure crop: {exc}", element_id=item.element_id)


# ---------------------------------------------------------------- pass 3


def _is_heading(item: _Item, body_size: float) -> tuple[bool, str]:
    text = clean_text(item.raw_text)
    if not text or len(text) > 150 or len(item.lines) > 3 or not re.search(r"[^\W\d_]{2,}", text):
        return False, ""
    max_size = max(l.size for l in item.lines)
    if max_size >= body_size + 2:
        return True, "font_size"
    if all(l.bold for l in item.lines) and NUMBERED_HEADING.match(text) and len(text) <= 100:
        return True, "numbered_bold"
    return False, ""


def _assemble(doc: ExtractedDocument, page_items: list[tuple[_Page, list[_Item]]], body_size: float) -> None:
    # Level = font-size rank (largest first); numbering depth only orders headings
    # of the same size ("1." above "1.3"). The result is re-ranked to 1, 2, 3...
    heading_info = {}
    for _, items in page_items:
        for item in items:
            if item.kind == "text":
                is_heading, how = _is_heading(item, body_size)
                if is_heading:
                    size = round(max(l.size for l in item.lines) * 2) / 2
                    numbered = NUMBERED_HEADING.match(clean_text(item.raw_text))
                    depth = numbered.group(1).count(".") if numbered else 0
                    heading_info[item.element_id] = (how, size, depth)
    sizes = sorted({s for _, s, _ in heading_info.values()}, reverse=True)
    raw_levels = {k: (sizes.index(s), d) for k, (_, s, d) in heading_info.items()}
    dense = {lv: i + 1 for i, lv in enumerate(sorted(set(raw_levels.values())))}
    levels = {k: dense[v] for k, v in raw_levels.items()}

    stack: list[tuple[int, str]] = []
    order = 0
    for layout, items in page_items:
        for item in items:
            path = [title for _, title in stack]
            kind = item.kind
            level = None
            if item.element_id in heading_info:
                text = clean_text(item.raw_text)
                level = levels[item.element_id]
                kind = "heading"
                while stack and stack[-1][0] >= level:
                    stack.pop()
                path = [title for _, title in stack]
                stack.append((level, text))

            if kind in ("text", "heading"):
                element_id = item.element_id.replace(":x", ":h") if kind == "heading" else item.element_id
                element = Element(
                    element_id=element_id, kind=kind, order=order,
                    text=clean_text(item.raw_text), raw_text=item.raw_text,
                    page=layout.number, page_label=layout.label, bbox=item.bbox,
                    section_path=path, heading_level=level,
                    location={"page": layout.number, "block": item.block_no,
                              **({"heading_source": heading_info[item.element_id][0]} if level else {})},
                )
            else:
                caption = item.caption
                text = _table_text(item.table) if item.table else (item.figure.text_labels or "")
                element = Element(
                    element_id=item.element_id, kind=kind, order=order,
                    text=clean_text(" ".join(p for p in (caption.text if caption else "", text) if p)) if kind == "figure" else text,
                    page=layout.number, page_label=layout.label, bbox=item.bbox,
                    section_path=path, label=caption.label if caption else None,
                    caption=caption.text if caption else None, caption_match=item.caption_match,
                    location={"page": layout.number, **item.location},
                    table=item.table, figure=item.figure,
                )
            doc.elements.append(element)
            order += 1


def _table_text(table: TableData) -> str:
    return "\n".join(" | ".join(c.text if c else "" for c in row) for row in table.normalised_cells)


def _continuation_hints(doc: ExtractedDocument, page_items: list[tuple[_Page, list[_Item]]]) -> None:
    """Flag tables that may continue onto the next page. Never merges them."""
    for (prev, prev_items), (nxt, next_items) in zip(page_items, page_items[1:]):
        if nxt.number != prev.number + 1:
            continue
        prev_tables = [i for i in prev_items if i.kind == "table"]
        next_tables = [i for i in next_items if i.kind == "table"]
        if not prev_tables or not next_tables:
            continue
        last, first = prev_tables[-1], next_tables[0]
        ends_low = last.bbox[3] > prev.rect.y0 + prev.rect.height * 0.75
        starts_high = first.bbox[1] < nxt.rect.y0 + nxt.rect.height * 0.25
        if not (ends_low and starts_high):
            continue
        prev_header = [c.text for c in last.table.normalised_cells[0] if c and c.text]
        next_header = [c.text for c in first.table.normalised_cells[0] if c and c.text]
        same_cols = last.table.n_cols == first.table.n_cols
        same_header = bool(prev_header) and prev_header == next_header
        if same_cols and same_header:
            evidence = "same column count and identical first row"
        elif same_header:
            evidence = "identical first-row text but different column count"
        elif same_cols:
            evidence = "same column count only"
        else:
            continue
        doc.add_issue("possible_table_continuation",
                      f"Table may continue across pages {prev.number}-{nxt.number} ({evidence}); kept as separate tables",
                      severity="info", element_ids=[last.element_id, first.element_id], evidence=evidence)


# ---------------------------------------------------------------- entry point


def parse_pdf(path: Path, doc: ExtractedDocument, out_dir: Path | None = None,
              pages: list[int] | None = None) -> ExtractedDocument:
    pdf = pymupdf.open(path)
    try:
        if pdf.needs_pass:
            doc.status = "failed"
            doc.add_issue("encrypted", "PDF is password protected", severity="error")
            return doc
        doc.page_count = pdf.page_count
        doc.metadata = {k: v for k, v in (pdf.metadata or {}).items() if v}
        toc = pdf.get_toc()
        if toc:
            doc.metadata["bookmarks"] = [{"level": lvl, "title": title, "page": p} for lvl, title, p in toc]

        requested = pages or list(range(1, pdf.page_count + 1))
        invalid = [p for p in requested if not 1 <= p <= pdf.page_count]
        if invalid:
            doc.add_issue("invalid_pages", f"Requested pages outside 1-{pdf.page_count}: {invalid}")
        page_numbers = sorted({p for p in requested if 1 <= p <= pdf.page_count})
        doc.pages_processed = page_numbers
        short = doc.doc_id[:10]

        layouts: list[_Page] = []
        size_counter: Counter = Counter()
        for pno in page_numbers:
            page = pdf[pno - 1]
            try:
                blocks = _read_blocks(page)
            except Exception as exc:
                doc.add_issue("page_text_failed", f"Text extraction failed: {exc}", severity="error", page=pno)
                blocks = []
            tables = []
            try:
                found = page.find_tables().tables
            except Exception as exc:
                doc.add_issue("table_detection_failed", f"Table detection failed: {exc}", severity="error", page=pno)
                found = []
            for i, table in enumerate(found):
                try:
                    tables.append(_table_item(page, table, i, f"{short}:p{pno}:tab{i + 1}"))
                except Exception as exc:
                    doc.add_issue("table_extraction_failed", f"Table {i + 1} could not be read: {exc}",
                                  severity="error", page=pno)
            layouts.append(_Page(pno, page.rect, blocks, tables))
            for block in blocks:
                for line in block.lines:
                    size_counter[line.size] += len(line.text)

        repeated, doc.furniture = _detect_furniture(layouts)
        body_size = size_counter.most_common(1)[0][0] if size_counter else 10.0
        doc.metadata["body_font_size"] = body_size

        counts: Counter = Counter()
        page_items = []
        for layout in layouts:
            page = pdf[layout.number - 1]
            doc.pages.append({"page": layout.number, "label": layout.label, "width": round(layout.rect.width, 1),
                              "height": round(layout.rect.height, 1), "rotation": page.rotation})
            try:
                page_items.append((layout, _build_page(page, layout, repeated, doc, out_dir, counts)))
            except Exception as exc:
                doc.add_issue("page_failed", f"Page could not be processed: {exc}", severity="error",
                              page=layout.number)

        for code, n in counts.items():
            doc.add_issue(code, f"{n} image(s)/region(s) not treated as figures ({code.replace('_', ' ')})",
                          severity="info", count=n)
        _assemble(doc, page_items, body_size)
        _continuation_hints(doc, page_items)

        uncaptioned = [e.element_id for e in doc.elements if e.kind in ("table", "figure") and not e.caption]
        if uncaptioned:
            doc.add_issue("objects_without_caption", f"{len(uncaptioned)} table(s)/figure(s) have no caption",
                          severity="info", element_ids=uncaptioned)
        return doc
    finally:
        pdf.close()
