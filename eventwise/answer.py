"""Grounded answers over retrieved evidence.

One chat completion per question. The model may cite only element IDs that were
placed in the prompt, and every rendered citation comes from retrieval metadata:
a result uses ``RetrievedEvidence.citation``; a context item uses the primary
index unit (document, page, section, label, location) via ``format_citation``.
A context item with no indexed unit is not citable.

Figure images are attached only when the question asks about a figure, chart,
graph, plot, diagram, image, or photo, or when a retrieved figure — or a figure
cited by a retrieved element — has no usable text and no extracted chart data.
At most two images are sent. Run the CLI from the project root so image paths
resolve.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .retrieve import RetrievalResponse, RetrievedEvidence, format_citation

DEFAULT_MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "none"
MAX_IMAGES = 2
TABLE_CHAR_LIMIT = 6000
TABLE_TRUNCATION_NOTE = "…(table truncated)"
ABSTAIN = "The documents do not contain enough information to answer this."

_VISUAL_QUERY = re.compile(
    r"\b(figures?|figs?\.|charts?|graphs?|plots?|diagrams?|images?|photos?)\b",
    re.IGNORECASE,
)
_CHART_DATA = re.compile(r"^Chart data\b", re.MULTILINE)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_SYSTEM = (
    "You answer questions about documents using only the evidence in the user message "
    "and any images attached for a named element id. If that evidence does not contain "
    "the answer, set sufficient to false and say what is missing. Do not use outside "
    "knowledge or invent numbers, dates, or sources. Return a JSON object with keys "
    "answer (string), citations (list of element_id strings copied from the evidence), "
    "and sufficient (boolean)."
)


@dataclass
class Citation:
    element_id: str
    citation: str
    kind: str


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    sufficient: bool
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Image:
    element_id: str
    path: str
    data_url: str


def _primary_unit(retriever, element_id: str):
    index = retriever.primary.get(element_id)
    if index is None:
        return None
    units = retriever.units
    if index < 0 or index >= len(units):
        return None
    return units[index]


def citation_from_index(retriever, element_id: str) -> Citation | None:
    """Citation for a context element from its primary index unit, or None if that unit is absent."""
    unit = _primary_unit(retriever, element_id)
    if unit is None or not (unit.doc_name or "").strip():
        return None
    part = unit.part if unit.rows is not None else None
    text = format_citation(
        unit.doc_name, unit.page, unit.page_label, list(unit.section_path or []),
        unit.label, dict(unit.location or {}), part,
    )
    if not text.strip():
        return None
    return Citation(element_id=element_id, citation=text, kind=unit.kind)


def allowed_citations(response: RetrievalResponse, retriever) -> dict[str, Citation]:
    """IDs the model may cite, in prompt order. Result citations win over a context lookup."""
    allowed: dict[str, Citation] = {}
    for result in response.results:
        allowed[result.element_id] = Citation(result.element_id, result.citation, result.kind)
    for result in response.results:
        for item in result.context:
            if item.element_id in allowed:
                continue
            citation = citation_from_index(retriever, item.element_id)
            if citation is not None:
                allowed[item.element_id] = citation
    return allowed


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, dict):
        return " ".join((cell.get("text") or "").split())
    return " ".join(str(cell).split())


def format_table_text(element: dict[str, Any]) -> str:
    """Pipe-separated grid. Empty cells are omitted, matching indexed table text."""
    table = element.get("table") or {}
    grid = table.get("normalised_cells") or table.get("raw_cells") or []
    lines = [" | ".join(text for text in (_cell_text(cell) for cell in row) if text) for row in grid]
    text = "\n".join(line for line in lines if line).strip()
    if len(text) <= TABLE_CHAR_LIMIT:
        return text
    cut = text.rfind("\n", 0, TABLE_CHAR_LIMIT)
    if cut < TABLE_CHAR_LIMIT // 2:
        cut = TABLE_CHAR_LIMIT
    return text[:cut].rstrip() + "\n" + TABLE_TRUNCATION_NOTE


def _load_table(retriever, element_id: str) -> tuple[str | None, str | None]:
    try:
        element = retriever.get_element(element_id)
    except KeyError:
        return None, f"Full table for {element_id} could not be loaded."
    text = format_table_text(element)
    if not text:
        return None, None
    warning = None
    if TABLE_TRUNCATION_NOTE in text:
        warning = f"Table {element_id} was truncated to {TABLE_CHAR_LIMIT} characters."
    return text, warning


def _cap_text(text: str) -> str:
    text = text.strip()
    if len(text) <= TABLE_CHAR_LIMIT:
        return text
    cut = text.rfind("\n", 0, TABLE_CHAR_LIMIT)
    if cut < TABLE_CHAR_LIMIT // 2:
        cut = TABLE_CHAR_LIMIT
    return text[:cut].rstrip() + "\n" + TABLE_TRUNCATION_NOTE


def table_bodies(retriever, result: RetrievedEvidence) -> tuple[str, dict[str, str], list[str]]:
    """Text to show for a hit, plus continuation grids keyed by element id."""
    warnings: list[str] = []
    continuations: dict[str, str] = {}
    if result.kind != "table":
        body = result.content.strip() or "(no text)"
    else:
        text, warning = _load_table(retriever, result.element_id)
        if warning:
            warnings.append(warning)
        if text:
            body = text
            caption = (result.caption or "").strip()
            if caption and caption not in text:
                body = caption + "\n" + body
            parts = [p.get("part") for p in result.matched_parts if p.get("part")]
            if parts:
                body = "Matched " + ", ".join(parts) + "\n" + body
        else:
            body = _cap_text(result.content) or "(no text)"
    for item in result.context:
        if item.relation != "table_continuation" or item.kind != "table":
            continue
        if item.element_id == result.element_id:
            continue
        text, warning = _load_table(retriever, item.element_id)
        if warning:
            warnings.append(warning)
        if text:
            continuations[item.element_id] = text
    return body, continuations, warnings


def _visual_question(query: str) -> bool:
    return _VISUAL_QUERY.search(query) is not None


def _has_chart_data(content: str) -> bool:
    return _CHART_DATA.search(content or "") is not None


def figure_image_candidates(query: str, results: list[RetrievedEvidence], retriever) -> list[tuple[str, str]]:
    """(element_id, image_path) that need vision, in result order, before the file and count limits."""
    visual = _visual_question(query)
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(element_id: str, path: str | None, content: str, low_text: bool) -> None:
        if not path or element_id in seen or _has_chart_data(content):
            return
        if not (low_text or visual):
            return
        seen.add(element_id)
        chosen.append((element_id, path))

    for result in results:
        if result.kind == "figure":
            add(result.element_id, result.image_path, result.content, result.low_text)
    for result in results:
        for item in result.context:
            if item.kind != "figure" or item.relation != "cites":
                continue
            unit = _primary_unit(retriever, item.element_id)
            if unit is None:
                continue
            add(item.element_id, item.image_path, unit.content, unit.low_text)
    return chosen


def load_images(candidates: list[tuple[str, str]], limit: int = MAX_IMAGES) -> tuple[list[_Image], list[str]]:
    loaded: list[_Image] = []
    warnings: list[str] = []
    for element_id, path in candidates:
        if len(loaded) >= limit:
            warnings.append(f"Further figure images were not sent (limit {limit}).")
            break
        file = Path(path)
        if not file.is_file():
            warnings.append(f"Figure image for {element_id} is missing ({path}).")
            continue
        encoded = base64.b64encode(file.read_bytes()).decode("ascii")
        loaded.append(_Image(element_id, path, f"data:image/png;base64,{encoded}"))
    return loaded, warnings


def _render_context(item, allowed: dict[str, Citation], continuations: dict[str, str]) -> str:
    if item.result_rank is not None:
        return f"- [{item.element_id}] {item.relation} (see evidence [{item.element_id}])"
    source = allowed.get(item.element_id)
    if source is None:
        header = (f"- [{item.element_id}] {item.relation}, {item.kind}. "
                  "Source location unavailable; do not cite this id.")
    else:
        header = f"- [{item.element_id}] {item.relation}, {item.kind}. Source: {source.citation}"
    lines = [header]
    snippet = (item.snippet or "").strip()
    if snippet:
        lines.append(f"  {snippet}")
    extra = continuations.get(item.element_id)
    if extra:
        lines.append("  Full table:")
        lines.append(extra)
    return "\n".join(lines)


def render_evidence(response: RetrievalResponse, retriever, allowed: dict[str, Citation]) -> tuple[str, list[str]]:
    blocks: list[str] = []
    warnings: list[str] = []
    for result in response.results:
        body, continuations, table_warnings = table_bodies(retriever, result)
        warnings.extend(table_warnings)
        lines = [
            f"[{result.element_id}] {result.kind}",
            f"Source: {result.citation}",
            "Content:",
            body,
        ]
        related = [
            _render_context(item, allowed, continuations)
            for item in result.context
            if item.element_id != result.element_id
        ]
        if related:
            lines.append("Related:")
            lines.extend(related)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), warnings


def build_user_text(query: str, evidence: str, image_ids: list[str], withheld: bool, unreadable: bool) -> str:
    parts = [f"Question: {query}", "", "Evidence:", evidence or "(none)"]
    if image_ids:
        listed = ", ".join(image_ids)
        parts.append("")
        parts.append(f"Attached images, in order: {listed}. No other figure was shown.")
    if withheld:
        parts.append("")
        parts.append("Figure images were not provided. If the answer is visible only in a figure, "
                     "set sufficient to false.")
    if unreadable:
        parts.append("")
        parts.append("Qualifying figure images could not be read. If the answer is visible only in a figure, "
                     "set sufficient to false.")
    return "\n".join(parts)


def user_content(text: str, images: list[_Image]) -> str | list[dict[str, Any]]:
    if not images:
        return text
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        content.append({"type": "text", "text": f"Image for element {image.element_id}:"})
        content.append({"type": "image_url", "image_url": {"url": image.data_url, "detail": "high"}})
    return content


def parse_model_json(text: str) -> dict[str, Any] | None:
    body = _JSON_FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_answer(data: dict[str, Any], allowed: dict[str, Citation]) -> tuple[str, bool, list[Citation], list[str]]:
    warnings: list[str] = []
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = ABSTAIN
        warnings.append("The model response did not include an answer.")
    else:
        answer = answer.strip()

    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        sufficient = False
        warnings.append("The model response did not include a boolean sufficient flag.")

    raw_ids = data.get("citations", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
        warnings.append("The model response did not include a citation list.")

    kept: list[Citation] = []
    seen: set[str] = set()
    rejected: list[str] = []
    for item in raw_ids:
        if not isinstance(item, str) or not item.strip():
            rejected.append(repr(item))
            continue
        element_id = item.strip()
        if element_id.startswith("[") and element_id.endswith("]"):
            element_id = element_id[1:-1].strip()
        if element_id in seen:
            continue
        seen.add(element_id)
        citation = allowed.get(element_id)
        if citation is None:
            rejected.append(element_id)
            continue
        kept.append(citation)
    if rejected:
        warnings.append("Dropped citation ids that were not in the supplied evidence: " + ", ".join(rejected))
    if sufficient and not kept:
        sufficient = False
        warnings.append("No valid citations remained, so the answer is not treated as sufficiently supported.")
    return answer, sufficient, kept, warnings


def _abstain(query: str, warnings: list[str], *, model: str | None = None,
             extra: str | None = None) -> GroundedAnswer:
    if extra:
        warnings.append(extra)
    return GroundedAnswer(query=query, answer=ABSTAIN, sufficient=False, warnings=warnings, model=model)


def answer_question(retriever, query: str, client, *, top_k: int = 10,
                    documents: Iterable[str] | None = None, vision: bool = True,
                    model: str = DEFAULT_MODEL) -> GroundedAnswer:
    """Retrieve evidence and ask the model for one grounded JSON answer."""
    response = retriever.search(query, top_k=top_k, documents=documents, mode="hybrid")
    warnings = list(response.warnings)
    if not response.results:
        return _abstain(query, warnings)

    allowed = allowed_citations(response, retriever)
    evidence, evidence_warnings = render_evidence(response, retriever, allowed)
    warnings.extend(evidence_warnings)
    candidates = figure_image_candidates(query, response.results, retriever)
    images: list[_Image] = []
    if vision:
        images, image_warnings = load_images(candidates)
        warnings.extend(image_warnings)
    text = build_user_text(
        query, evidence, [image.element_id for image in images],
        withheld=not vision and bool(candidates),
        unreadable=vision and bool(candidates) and not images,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content(text, images)},
        ],
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT,
        max_completion_tokens=1000,
        store=False,
    )
    raw = completion.choices[0].message.content or ""
    data = parse_model_json(raw) if isinstance(raw, str) else None
    image_ids = [image.element_id for image in images]
    if data is None:
        return _abstain(query, warnings, model=model, extra="The model response was not valid JSON.")
    answer, sufficient, citations, answer_warnings = validate_answer(data, allowed)
    warnings.extend(answer_warnings)
    return GroundedAnswer(
        query=query, answer=answer, sufficient=sufficient, citations=citations,
        warnings=warnings, model=model, images=image_ids,
    )
