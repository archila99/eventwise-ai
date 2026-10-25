"""Caption detection and caption-to-object association.

Captions are recognised by a generic label pattern ("Figure 3:", "Table 2.1",
"Tabl 1:", "Ffigur 2"), not by any document-specific layout. Associations are
scored and reported with a confidence so uncertain matches stay visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import BBox

FIGURE_WORDS = ("Figure", "FIGURE", "Fig.", "Fig", "Chart", "Graph", "Diagram", "Exhibit",
                "Illustration", "Map", "Ffigur", "FFIGUR", "Siart", "Graff")
TABLE_WORDS = ("Table", "TABLE", "Tab.", "Tabl", "TABL")

_WORDS = "|".join(re.escape(w) for w in sorted(set(FIGURE_WORDS + TABLE_WORDS), key=len, reverse=True))
# The label must be followed by a separator or end the line, so running text
# such as "Table 4 is an example rota" is not treated as a caption.
_CAPTION = re.compile(
    rf"^(?P<word>{_WORDS})\s*(?P<num>[A-Z]?\d+(?:[.\-]\d+)*[a-z]?)(?:\s*[:.\u2013\u2014-]\s*(?P<rest>.*)|\s*$)",
    re.DOTALL,
)

_MENTION = re.compile(rf"(?<![\w.])(?P<word>{_WORDS})\s*(?P<num>[A-Z]?\d+(?:[.\-]\d+)*[a-z]?)(?![\w]|[.\-]\d)")

MAX_CAPTION_DISTANCE_PT = 72.0  # one inch


def _key(word: str, num: str) -> str:
    return f"{'table' if word in TABLE_WORDS else 'figure'}:{num.lower()}"


def label_key(label: str) -> str | None:
    """Comparable key for a caption label: 'Figure 1.1.2' -> 'figure:1.1.2', 'Tabl 3' -> 'table:3'."""
    match = _MENTION.match(label.strip())
    return _key(match["word"], match["num"]) if match else None


def find_label_mentions(text: str) -> set[str]:
    """Keys of all table/figure labels mentioned anywhere in ``text``."""
    return {_key(m["word"], m["num"]) for m in _MENTION.finditer(text)}


def mention_offsets(text: str, key: str) -> list[int]:
    """Character offsets where the label with ``key`` is mentioned."""
    return [m.start() for m in _MENTION.finditer(text) if _key(m["word"], m["num"]) == key]


@dataclass
class Caption:
    kind: str  # "figure" | "table"
    label: str  # e.g. "Table 3"
    text: str  # full caption text


def parse_caption(text: str) -> Caption | None:
    match = _CAPTION.match(text.strip())
    if not match:
        return None
    word = match["word"]
    kind = "table" if word in TABLE_WORDS else "figure"
    return Caption(kind=kind, label=f"{word} {match['num']}", text=text.strip())


def box_distance(a: BBox, b: BBox) -> float:
    """Shortest distance between two rectangles (0 if they overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class Assignment:
    caption_index: int
    object_index: int
    confidence: str
    distance: float
    method: str


def match_by_geometry(
    captions: list[tuple[Caption, BBox]],
    objects: list[tuple[str, BBox]],
    max_distance: float = MAX_CAPTION_DISTANCE_PT,
) -> list[Assignment]:
    """One-to-one caption/object matching on a single page.

    Same-kind pairs are preferred; captions below or beside an object are
    preferred over captions above it (the commonest convention), but all
    placements are allowed.
    """
    candidates = []
    for ci, (cap, cbox) in enumerate(captions):
        for oi, (okind, obox) in enumerate(objects):
            dist = box_distance(cbox, obox)
            if dist > max_distance:
                continue
            mismatch = cap.kind != okind
            above = cbox[3] <= obox[1] + 2  # caption sits entirely above the object
            score = dist + (100 if mismatch else 0) + (8 if above else 0)
            candidates.append((score, ci, oi, dist, mismatch))
    candidates.sort()

    same_kind_distances: dict[int, list[float]] = {}
    for _, ci, _, dist, mismatch in candidates:
        if not mismatch:
            same_kind_distances.setdefault(ci, []).append(dist)

    used_c, used_o, result = set(), set(), []
    for _, ci, oi, dist, mismatch in candidates:
        if ci in used_c or oi in used_o:
            continue
        used_c.add(ci)
        used_o.add(oi)
        others = sorted(same_kind_distances.get(ci, []))[1:]
        clear_winner = not others or others[0] >= 2 * max(dist, 6.0)
        if mismatch:
            confidence = "low"
        elif dist <= 24 and clear_winner:
            confidence = "high"
        else:
            confidence = "medium"
        result.append(Assignment(ci, oi, confidence, round(dist, 1), "geometry"))
    return result


def match_by_adjacency(sequence: list[tuple[str, Caption | None]]) -> tuple[list[Assignment], list[int]]:
    """Match captions to directly neighbouring tables/figures in reading order.

    ``sequence`` holds (kind, caption) for each element in order, where kind is
    "table", "figure", "caption" or "other". Returns assignments and the
    indices of captions that had two equally plausible neighbours.
    """
    used, result, ambiguous = set(), [], []
    for i, (kind, cap) in enumerate(sequence):
        if kind != "caption" or cap is None:
            continue
        neighbours = []
        for j, placement in ((i - 1, "after_object"), (i + 1, "before_object")):
            if 0 <= j < len(sequence) and sequence[j][0] in ("table", "figure") and j not in used:
                neighbours.append((j, placement, sequence[j][0] == cap.kind))
        same_kind = [n for n in neighbours if n[2]]
        if same_kind:
            pick = same_kind[0]  # prefer caption-below-object (previous element)
            confidence = "high" if len(same_kind) == 1 else "medium"
            if len(same_kind) > 1:
                ambiguous.append(i)
        elif neighbours:
            pick, confidence = neighbours[0], "low"
        else:
            continue
        used.add(pick[0])
        result.append(Assignment(i, pick[0], confidence, 0.0, f"adjacency:{pick[1]}"))
    return result, ambiguous
