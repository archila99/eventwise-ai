"""Tunable settings for indexing and retrieval (no document-specific values)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

DEFAULT_EXTRACTED_DIR = ".cache/extracted"
DEFAULT_INDEX_DIR = ".cache/index"
DEFAULT_MODEL_DIR = ".cache/models"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class IndexConfig:
    # Text longer than this is split into overlapping windows. MiniLM reads at most
    # 256 word pieces (~180 words), so longer units would be silently truncated.
    max_unit_words: int = 160
    chunk_overlap_words: int = 30
    # Tables with more data rows (or words) than this also get row-group units.
    table_rows_threshold: int = 8
    table_words_threshold: int = 160
    table_group_rows: int = 6
    table_group_max_words: int = 140
    # Context borrowed for figures/tables from referencing or nearby text.
    context_words: int = 45
    max_context_snippets: int = 3
    figure_label_words: int = 120

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    candidate_k: int = 100  # per-method candidate list fed into fusion
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    # Each method's best hits survive fusion: RRF otherwise prefers items that are
    # mediocre in both rankings over an item ranked near the top by only one.
    guaranteed_per_method: int = 3
    # Soft document diversity: the n-th further result from the same document has its
    # score multiplied by decay**n while selecting (1.0 = off). Never excludes a document.
    same_document_decay: float = 0.85
    # Hard cap per document (None = no cap).
    max_per_document: int | None = None
    near_duplicate_cosine: float = 0.97
    heading_weight: float = 1.0  # <1 ranks headings lower; evaluation showed no benefit, so off by default
    navigation_penalty: float = 0.5  # multiplier for table-of-contents lines (kept, but ranked lower)
    suppress_covered_headings: bool = True
    expand_neighbours: bool = True
    neighbour_window: int = 1
    max_context_items: int = 6

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RetrievalConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown retrieval settings: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
