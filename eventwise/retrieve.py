"""Hybrid evidence retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion.

Pipeline per query:
1. filter units by document and element kind;
2. rank units with BM25 (exact terms, IDs, numbers) and with cosine similarity;
3. fuse the two rankings with RRF: score = sum_i w_i / (k + rank_i);
4. collapse units to elements (a table hit through several row groups is one
   result that lists the matched rows);
5. skip duplicate evidence (identical or near-identical content, headings whose
   section is already represented) and optionally cap results per document;
6. attach related elements (cited figures/tables, citing text, neighbours,
   cross-page table partners) as context without re-ranking them.

Command line:
    python -m eventwise.retrieve "question" [-k 10] [--doc NAME] [--kind table] [--mode hybrid|bm25|dense]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .config import DEFAULT_INDEX_DIR, RetrievalConfig
from .embeddings import Embedder, cosine_scores
from .index import IndexStore, LoadedIndex
from .tokenize import tokenize
from .units import IndexUnit

KINDS = ("heading", "text", "table", "figure")
MODES = ("hybrid", "bm25", "dense")


class KeywordIndex:
    """BM25 over unit search texts (built in memory from the stored units)."""

    def __init__(self, texts: Sequence[str]):
        self.n = len(texts)
        tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokens) if self.n else None

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        if not self._bm25 or not terms:
            return np.zeros(self.n, dtype=np.float64)
        return np.asarray(self._bm25.get_scores(terms), dtype=np.float64)


def top_indices(scores: np.ndarray, mask: np.ndarray, k: int, min_score: float | None = None) -> list[int]:
    """Indices of the k highest scores within ``mask`` (ties broken by index for determinism)."""
    candidates = np.flatnonzero(mask if min_score is None else mask & (scores > min_score))
    if candidates.size == 0:
        return []
    order = np.lexsort((candidates, -scores[candidates]))
    return [int(i) for i in candidates[order[:k]]]


def reciprocal_rank_fusion(rankings: dict[str, list[int]], weights: dict[str, float] | None = None,
                           k: int = 60) -> dict[int, float]:
    """RRF over ranked lists of item IDs (rank 1 = first). Items absent from a list get nothing from it."""
    fused: dict[int, float] = {}
    for name, ranking in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return fused


@dataclass
class ContextItem:
    element_id: str
    kind: str
    relation: str  # cites | cited_by | table_continuation | previous | next | context_source
    page: int | None
    page_label: str | None
    label: str | None
    caption: str | None
    snippet: str
    image_path: str | None = None
    result_rank: int | None = None  # set when the item is itself one of the ranked results


@dataclass
class RetrievedEvidence:
    rank: int
    element_id: str
    doc_id: str
    doc_name: str
    source_path: str
    kind: str
    score: float  # fused RRF score (or the single method's score in bm25/dense mode)
    selection_score: float  # score after the same-document decay; this is what orders the results
    bm25_rank: int | None
    bm25_score: float | None
    dense_rank: int | None
    dense_score: float | None
    citation: str
    page: int | None
    page_label: str | None
    section_path: list[str]
    location: dict[str, Any]
    label: str | None
    caption: str | None
    caption_confidence: str | None
    content: str  # the best-matching unit (e.g. the matched table rows with caption and header)
    matched_parts: list[dict[str, Any]]
    image_path: str | None
    element_ref: dict[str, str]  # where to load the complete original element
    low_text: bool = False
    selection: str = "fused"  # or "guaranteed (top 3 of bm25)" when kept by the per-method guarantee
    also_in: list[dict[str, Any]] = field(default_factory=list)  # duplicates that were not listed separately
    context: list[ContextItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResponse:
    query: str
    mode: str
    results: list[RetrievedEvidence]
    filters: dict[str, Any]
    config: dict[str, Any]
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    suppressed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_citation(doc_name: str, page: int | None, page_label: str | None, section_path: list[str],
                    label: str | None, location: dict[str, Any], part: str | None = None) -> str:
    bits = [doc_name]
    if page is not None:
        bits.append(f"p. {page}" + (f" (printed {page_label})" if page_label and page_label != str(page) else ""))
    else:
        loc = []
        for key in ("container", "table", "row", "col", "paragraph", "block"):
            if key in location and location[key] is not None:
                loc.append(f"{key} {location[key]}")
        if loc:
            bits.append(", ".join(loc[:3]))
    if section_path:
        bits.append("§ " + section_path[-1])
    if label:
        bits.append(label)
    if part:
        bits.append(part)
    return " — ".join(bits)


def _norm_content(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class HybridRetriever:
    def __init__(self, index: LoadedIndex, embedder: Embedder | None = None,
                 config: RetrievalConfig | None = None):
        self.index = index
        self.config = config or RetrievalConfig()
        self.embedder = embedder
        self.units: list[IndexUnit] = index.units
        self.vectors = index.vectors
        self.warnings: list[str] = list(index.problems)
        if embedder is not None and index.embedder_name and embedder.name != index.embedder_name:
            self.warnings.append(f"Query embedder {embedder.name} differs from index embedder "
                                 f"{index.embedder_name}; dense retrieval disabled. Re-run indexing.")
            self.embedder = None
        start = time.perf_counter()
        self.keyword = KeywordIndex([u.search_text for u in self.units])
        self.build_seconds = time.perf_counter() - start

        self.kind_arr = np.array([u.kind for u in self.units])
        self.doc_arr = np.array([u.doc_id for u in self.units])
        self.by_element: dict[str, list[int]] = {}
        for i, u in enumerate(self.units):
            self.by_element.setdefault(u.element_id, []).append(i)
        self.primary: dict[str, int] = {eid: next((i for i in idx if self.units[i].is_primary), idx[0])
                                        for eid, idx in self.by_element.items()}
        self.doc_order: dict[str, list[str]] = {}
        for eid, i in sorted(self.primary.items(), key=lambda kv: (self.units[kv[1]].doc_id, self.units[kv[1]].order)):
            self.doc_order.setdefault(self.units[i].doc_id, []).append(eid)
        self.position = {eid: n for ids in self.doc_order.values() for n, eid in enumerate(ids)}
        self._documents_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_disk(cls, index_dir: str | Path = DEFAULT_INDEX_DIR, embedder: Embedder | None = None,
                  config: RetrievalConfig | None = None) -> "HybridRetriever":
        return cls(IndexStore(index_dir).load(), embedder, config)

    # ----------------------------------------------------------------- filters
    def resolve_documents(self, documents: Iterable[str] | None) -> tuple[set[str] | None, list[str]]:
        """Doc IDs matching each filter (exact ID, exact name, or case-insensitive substring of name/path)."""
        if not documents:
            return None, []
        selected: set[str] = set()
        warnings = []
        docs = self.index.documents
        for wanted in documents:
            w = wanted.lower()
            hits = {d for d, info in docs.items()
                    if d == wanted or info["doc_name"].lower() == w
                    or w in info["doc_name"].lower() or w in info["source_path"].lower()}
            if not hits:
                warnings.append(f"Document filter '{wanted}' matches no indexed document")
            selected |= hits
        return selected, warnings

    def _mask(self, doc_ids: set[str] | None, kinds: Iterable[str] | None) -> np.ndarray:
        mask = np.ones(len(self.units), dtype=bool)
        if doc_ids is not None:
            mask &= np.isin(self.doc_arr, list(doc_ids))
        if kinds:
            mask &= np.isin(self.kind_arr, list(kinds))
        return mask

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int | None = None, documents: Iterable[str] | None = None,
               kinds: Iterable[str] | None = None, mode: str = "hybrid",
               config: RetrievalConfig | None = None) -> RetrievalResponse:
        cfg = config or self.config
        top_k = top_k or cfg.top_k
        timings: dict[str, float] = {}
        start = time.perf_counter()
        warnings = list(self.warnings)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        kinds = list(kinds) if kinds else None
        bad = [k for k in kinds or [] if k not in KINDS]
        if bad:
            raise ValueError(f"unknown element kind(s) {bad}; choose from {KINDS}")
        doc_ids, doc_warnings = self.resolve_documents(documents)
        warnings += doc_warnings
        mask = self._mask(doc_ids, kinds)
        filters = {"documents": list(documents or []), "kinds": kinds or [], "matched_doc_ids": sorted(doc_ids or [])}

        if mode in ("hybrid", "dense") and self.embedder is None:
            if mode == "dense":
                raise RuntimeError("dense retrieval needs an embedder")
            warnings.append("No query embedder available: BM25 only")
            mode = "bm25"

        rankings: dict[str, list[int]] = {}
        bm25_scores = dense_scores = None
        if mode in ("hybrid", "bm25"):
            t = time.perf_counter()
            bm25_scores = self.keyword.scores(query)
            rankings["bm25"] = top_indices(bm25_scores, mask, cfg.candidate_k, min_score=0.0)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("hybrid", "dense"):
            t = time.perf_counter()
            qvec = self.embedder.encode([query])[0]
            timings["query_embedding"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense_scores = cosine_scores(qvec, self.vectors)
            rankings["dense"] = top_indices(dense_scores, mask, cfg.candidate_k)
            timings["dense"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(rankings, {"bm25": cfg.bm25_weight, "dense": cfg.dense_weight}, cfg.rrf_k)
        else:
            only = rankings[mode]
            raw = bm25_scores if mode == "bm25" else dense_scores
            fused = {i: float(raw[i]) for i in only}
        for i in fused:
            unit = self.units[i]
            if unit.navigation:
                fused[i] *= cfg.navigation_penalty
            if unit.kind == "heading":
                fused[i] *= cfg.heading_weight
        unit_rank = {name: {u: r for r, u in enumerate(rank, start=1)} for name, rank in rankings.items()}
        guaranteed: list[tuple[str, str]] = []  # (element_id, why)
        per_method = min(cfg.guaranteed_per_method, top_k // 3)  # never more than a third of the slots each
        if mode == "hybrid" and per_method > 0:
            for name, ranking in rankings.items():
                taken: list[str] = []
                for u in ranking:
                    eid = self.units[u].element_id
                    if self.units[u].kind == "heading" or self.units[u].navigation or eid in taken:
                        continue
                    taken.append(eid)
                    guaranteed.append((eid, f"top {per_method} of {name}"))
                    if len(taken) >= per_method:
                        break
        results, suppressed = self._select(fused, unit_rank, bm25_scores, dense_scores, top_k, cfg, guaranteed)
        timings["fusion_and_selection"] = (time.perf_counter() - t) * 1000
        if cfg.expand_neighbours:
            t = time.perf_counter()
            self._expand(results, cfg)
            timings["expansion"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - start) * 1000
        if not results:
            warnings.append("No evidence found for this query with the current filters")
        return RetrievalResponse(query=query, mode=mode, results=results, filters=filters, config=cfg.to_dict(),
                                 timings_ms={k: round(v, 2) for k, v in timings.items()}, warnings=warnings,
                                 suppressed=suppressed)

    def _select(self, fused, unit_rank, bm25_scores, dense_scores, top_k, cfg, guaranteed=()):
        # Collapse units to elements: an element scores as its best unit.
        elements: dict[str, dict[str, Any]] = {}
        for u_idx, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
            eid = self.units[u_idx].element_id
            entry = elements.setdefault(eid, {"score": score, "units": []})
            entry["units"].append(u_idx)
        remaining = sorted(elements, key=lambda e: -elements[e]["score"])

        results: list[RetrievedEvidence] = []
        chosen_vecs: list[tuple[RetrievedEvidence, np.ndarray, str]] = []
        seen_content: dict[tuple[str, str], RetrievedEvidence] = {}
        per_doc: dict[str, int] = {}
        suppressed: list[dict[str, Any]] = []
        rejected: set[str] = set()

        def best_rank(eid: str, name: str) -> tuple[int | None, float | None]:
            ranks = [(unit_rank[name][u], u) for u in self.by_element[eid] if u in unit_rank.get(name, {})]
            if not ranks:
                return None, None
            rank, u = min(ranks)
            raw = bm25_scores if name == "bm25" else dense_scores
            return rank, round(float(raw[u]), 4)

        def rejection(eid: str) -> tuple[str, RetrievedEvidence | None] | None:
            """Why ``eid`` cannot be added now (no side effects), or None."""
            primary = self.units[self.primary[eid]]
            norm = _norm_content(primary.content)
            if len(norm.split()) >= 8 and (primary.kind, norm) in seen_content:
                return "duplicate", seen_content[(primary.kind, norm)]
            if self.vectors.size and primary.kind != "heading":
                vec = self.vectors[self.primary[eid]]
                for chosen_result, other, kind in chosen_vecs:
                    if kind == primary.kind and float(vec @ other) >= cfg.near_duplicate_cosine:
                        return "duplicate", chosen_result
            if cfg.suppress_covered_headings and primary.kind == "heading":
                path = primary.section_path + [primary.content]
                if any(r.doc_id == primary.doc_id and r.section_path[:len(path)] == path for r in results):
                    return "heading of a section already in results", None
            if cfg.max_per_document and per_doc.get(primary.doc_id, 0) >= cfg.max_per_document:
                return f"max_per_document={cfg.max_per_document}", None
            return None

        def adjusted(eid: str) -> float:
            n = per_doc.get(self.units[self.primary[eid]].doc_id, 0)
            return elements[eid]["score"] * cfg.same_document_decay ** n

        def try_add(eid: str, selection: str) -> bool:
            entry = elements[eid]
            selection_score = adjusted(eid)
            best = self.units[entry["units"][0]]
            primary = self.units[self.primary[eid]]
            why = rejection(eid)
            if why is not None:
                reason, dup_of = why
                if dup_of is not None:
                    dup_of.also_in.append({"element_id": eid, "doc_name": best.doc_name, "page": best.page,
                                           "citation": self._citation(best)})
                suppressed.append({"element_id": eid, "reason": reason,
                                   **({"of": dup_of.element_id} if dup_of is not None else {})})
                return False
            key = (primary.kind, _norm_content(primary.content))
            b_rank, b_score = best_rank(eid, "bm25")
            d_rank, d_score = best_rank(eid, "dense")
            evidence = self._evidence(len(results) + 1, eid, entry, best, primary, b_rank, b_score, d_rank, d_score)
            evidence.selection = selection
            evidence.selection_score = round(selection_score, 6)
            results.append(evidence)
            per_doc[primary.doc_id] = per_doc.get(primary.doc_id, 0) + 1
            seen_content.setdefault(key, evidence)
            if self.vectors.size:
                chosen_vecs.append((evidence, self.vectors[self.primary[eid]], primary.kind))
            return True

        # Greedy selection; the soft decay lets other documents compete without excluding any.
        while remaining and len(results) < top_k:
            pick = max(remaining, key=lambda e: (adjusted(e), -remaining.index(e)))
            remaining.remove(pick)
            if not try_add(pick, "fused"):
                rejected.add(pick)

        # Each method's strongest hits must not be lost in fusion.
        chosen = {r.element_id for r in results}
        for eid, why in guaranteed:
            if eid in chosen or eid in rejected or eid not in elements:
                continue
            if rejection(eid) is not None:
                try_add(eid, "")  # records why it was skipped
                continue
            if len(results) >= top_k:
                drop = next((r for r in reversed(results) if not r.selection.startswith("guaranteed")), None)
                if drop is None:
                    break
                results.remove(drop)
                per_doc[drop.doc_id] -= 1
                chosen_vecs[:] = [c for c in chosen_vecs if c[0] is not drop]
                for k in [k for k, v in seen_content.items() if v is drop]:
                    del seen_content[k]
                suppressed.append({"element_id": drop.element_id, "reason": f"displaced by guaranteed {eid}"})
            if try_add(eid, f"guaranteed ({why})"):
                chosen.add(eid)
        for n, r in enumerate(results, start=1):
            r.rank = n
        return results, suppressed

    def _image(self, unit: IndexUnit) -> str | None:
        return str(self.index.extracted_dir / unit.image_path) if unit.image_path else None

    def _citation(self, unit: IndexUnit) -> str:
        part = unit.part if unit.rows is not None else None
        return format_citation(unit.doc_name, unit.page, unit.page_label, unit.section_path, unit.label,
                               unit.location, part)

    def _evidence(self, rank, eid, entry, best: IndexUnit, primary: IndexUnit, b_rank, b_score, d_rank, d_score):
        info = self.index.documents.get(best.doc_id, {})
        parts = [{"unit_id": self.units[u].unit_id, "part": self.units[u].part, "rows": self.units[u].rows}
                 for u in entry["units"] if self.units[u].part]
        return RetrievedEvidence(
            rank=rank, element_id=eid, doc_id=best.doc_id, doc_name=best.doc_name,
            source_path=info.get("source_path", ""), kind=best.kind, score=round(entry["score"], 6),
            selection_score=round(entry["score"], 6),
            bm25_rank=b_rank, bm25_score=b_score, dense_rank=d_rank, dense_score=d_score,
            citation=self._citation(best), page=best.page, page_label=best.page_label,
            section_path=best.section_path, location=best.location, label=best.label, caption=best.caption,
            caption_confidence=best.caption_confidence, content=best.content, matched_parts=parts,
            image_path=self._image(best), low_text=primary.low_text,
            element_ref={"document_json": str(self.index.extracted_dir / best.doc_id / "document.json"),
                         "element_id": eid},
        )

    def _expand(self, results: list[RetrievedEvidence], cfg: RetrievalConfig) -> None:
        ranked = {r.element_id: r.rank for r in results}
        for r in results:
            primary = self.units[self.primary[r.element_id]]
            related: list[tuple[str, str]] = []
            related += [(e, "cites") for e in primary.refs]
            related += [(e, "cited_by") for e in primary.cited_by]
            related += [(e, "table_continuation") for e in primary.continues]
            related += [(e, "context_source") for e in primary.context_sources]
            order = self.doc_order.get(primary.doc_id, [])
            pos = self.position.get(r.element_id)
            if pos is not None:
                for step in range(1, cfg.neighbour_window + 1):
                    for j, rel in ((pos - step, "previous"), (pos + step, "next")):
                        if 0 <= j < len(order):
                            other = self.units[self.primary[order[j]]]
                            if other.section_path == primary.section_path or other.kind == "heading":
                                related.append((order[j], rel))
            seen = {r.element_id}
            for eid, relation in related:
                if eid in seen or eid not in self.primary or len(r.context) >= cfg.max_context_items:
                    continue
                seen.add(eid)
                u = self.units[self.primary[eid]]
                snippet = u.content if len(u.content) <= 400 else u.content[:400] + " …"
                r.context.append(ContextItem(element_id=eid, kind=u.kind, relation=relation, page=u.page,
                                             page_label=u.page_label, label=u.label, caption=u.caption,
                                             snippet=snippet, image_path=self._image(u),
                                             result_rank=ranked.get(eid)))

    # ----------------------------------------------------------------- originals
    def get_element(self, element_id: str) -> dict[str, Any]:
        """The complete original element (full table grid, figure metadata) from ``document.json``."""
        idx = self.primary.get(element_id)
        if idx is None:
            raise KeyError(element_id)
        doc_id = self.units[idx].doc_id
        if doc_id not in self._documents_cache:
            path = self.index.extracted_dir / doc_id / "document.json"
            self._documents_cache[doc_id] = {e["element_id"]: e for e in json.loads(path.read_text())["elements"]}
        return self._documents_cache[doc_id][element_id]


# --------------------------------------------------------------------- CLI
def _print_response(resp: RetrievalResponse, show_context: bool, width: int) -> None:
    print(f'Query: "{resp.query}"   mode={resp.mode}   results={len(resp.results)}   '
          f"latency={resp.timings_ms.get('total', 0):.0f} ms")
    if resp.filters["documents"] or resp.filters["kinds"]:
        print(f"Filters: documents={resp.filters['documents']} kinds={resp.filters['kinds']}")
    for w in resp.warnings:
        print(f"WARNING: {w}")
    for r in resp.results:
        ranks = []
        if r.bm25_rank:
            ranks.append(f"bm25 #{r.bm25_rank} ({r.bm25_score:.2f})")
        if r.dense_rank:
            ranks.append(f"dense #{r.dense_rank} ({r.dense_score:.3f})")
        kept = f"  [{r.selection}]" if r.selection != "fused" else ""
        adj = f" (after doc decay {r.selection_score:.4f})" if abs(r.selection_score - r.score) > 1e-9 else ""
        print(f"\n[{r.rank}] {r.kind.upper():7} score={r.score:.4f}{adj}  {'  '.join(ranks)}{kept}")
        print(f"    {r.citation}")
        print(f"    id={r.element_id}" + (f"  caption match={r.caption_confidence}" if r.caption_confidence else ""))
        if r.matched_parts:
            print("    matched: " + ", ".join(p["part"] for p in r.matched_parts if p["part"]))
        if r.image_path:
            print(f"    image: {r.image_path}")
        if r.low_text:
            print("    note: element has little searchable text of its own")
        body = r.content.strip().replace("\n", " ⏎ ")
        print("    " + (body if len(body) <= width else body[:width] + " …"))
        if r.also_in:
            print("    also in: " + "; ".join(a["citation"] for a in r.also_in))
        if show_context:
            for c in r.context:
                tag = f" (= result {c.result_rank})" if c.result_rank else ""
                snippet = c.snippet.replace("\n", " ⏎ ")
                print(f"      ↳ {c.relation}: {c.kind} {c.label or ''} p.{c.page}{tag}: {snippet[:110]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve evidence for a question (no LLM).")
    parser.add_argument("query")
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("--doc", action="append", help="Restrict to documents (ID, name or substring); repeatable")
    parser.add_argument("--kind", action="append", choices=KINDS, help="Restrict to element kinds; repeatable")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--max-per-doc", type=int, default=None, help="Cap results per document (default: no cap)")
    parser.add_argument("--no-context", action="store_true", help="Do not show related context elements")
    parser.add_argument("--json", action="store_true", help="Print the structured response as JSON")
    parser.add_argument("--index", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--width", type=int, default=300, help="Characters of content to show per result")
    args = parser.parse_args(argv)

    from .embeddings import SentenceTransformerEmbedder
    from .index import IndexUnavailable

    t = time.perf_counter()
    try:
        index = IndexStore(args.index).load()
    except IndexUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    embedder = SentenceTransformerEmbedder(allow_download=False) if args.mode != "bm25" else None
    config = replace(RetrievalConfig(), max_per_document=args.max_per_doc)
    retriever = HybridRetriever(index, embedder, config)
    if embedder is not None:
        embedder.encode(["warm-up"])
    startup = time.perf_counter() - t
    resp = retriever.search(args.query, args.top_k, args.doc, args.kind, args.mode)
    if args.json:
        print(json.dumps(resp.to_dict(), ensure_ascii=False, indent=1))
    else:
        _print_response(resp, not args.no_context, args.width)
        print(f"\n(startup {startup:.1f}s: index load + BM25 build + model load; "
              f"{len(index.units)} units from {len(index.documents)} documents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
