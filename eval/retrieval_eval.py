"""Retrieval-only evaluation (no LLM).

For every question, each *required evidence item* counts separately, so a
multi-source question is only fully satisfied when every source is retrieved.

    python eval/retrieval_eval.py [--index .cache/index] [--out eval/results] [--skip-build-timing]

Reports Recall@5/@10 (ranked results only, and with neighbour context), results
by evidence type, BM25-only / dense-only / hybrid comparisons, query latency,
index build time, cache reuse and incremental indexing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eventwise.config import RetrievalConfig  # noqa: E402
from eventwise.embeddings import SentenceTransformerEmbedder  # noqa: E402
from eventwise.index import IndexStore, sync_index  # noqa: E402
from eventwise.retrieve import HybridRetriever  # noqa: E402

KS = (5, 10)


def resolve_locators(gold: dict, index) -> dict[tuple[str, int, int], set[str]]:
    """(question, item, alternative) -> element IDs satisfying the locator."""
    elements = []
    for doc_id in index.documents:
        doc = json.loads((index.extracted_dir / doc_id / "document.json").read_text())
        for e in doc["elements"]:
            elements.append((doc["doc_name"], e))
    resolved = {}
    for q in gold["questions"]:
        for i, item in enumerate(q["evidence"]):
            for j, loc in enumerate(item["any_of"]):
                ids = set()
                for name, e in elements:
                    text = (e.get("caption") or "") + "\n" + e["text"]
                    if (name == loc["doc"] and loc.get("page") in (None, e.get("page"))
                            and loc.get("kind") in (None, e["kind"]) and loc.get("label") in (None, e.get("label"))
                            and (loc.get("contains") is None or loc["contains"].lower() in text.lower())):
                        ids.add(e["element_id"])
                if not ids:
                    raise SystemExit(f"Gold locator resolves to nothing: {q['id']} item {i} {loc}")
                resolved[(q["id"], i, j)] = ids
    return resolved


def item_kind(retriever, ids: set[str]) -> str:
    kinds = {retriever.units[retriever.primary[e]].kind for e in ids if e in retriever.primary}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def evaluate(retriever, gold, resolved, mode: str, config: RetrievalConfig, label: str):
    rows, latencies = [], []
    for q in gold["questions"]:
        resp = retriever.search(q["question"], top_k=max(KS), documents=q.get("documents"), mode=mode, config=config)
        latencies.append(resp.timings_ms["total"])
        ranked = [r.element_id for r in resp.results]
        items = []
        for i, item in enumerate(q["evidence"]):
            ids = set().union(*(resolved[(q["id"], i, j)] for j in range(len(item["any_of"]))))
            rank = next((n for n, eid in enumerate(ranked, 1) if eid in ids), None)
            ctx_hit = {k: rank is not None and rank <= k or any(
                c.element_id in ids for r in resp.results[:k] for c in r.context) for k in KS}
            items.append({"name": item["name"], "kind": item_kind(retriever, ids), "rank": rank,
                          **{f"hit@{k}": rank is not None and rank <= k for k in KS},
                          **{f"ctx@{k}": ctx_hit[k] for k in KS}})
        top = resp.results[0] if resp.results else None
        rows.append({"id": q["id"], "priority": q.get("priority", False), "negative": q.get("negative", False),
                     "heldout": q.get("heldout", False),
                     "items": items, "latency_ms": resp.timings_ms["total"],
                     "top": top.citation if top else None, "top_score": top.score if top else None,
                     "top_dense": max((r.dense_score or 0) for r in resp.results) if resp.results else None,
                     "docs_in_top10": len({r.doc_id for r in resp.results})})
    return {"label": label, "mode": mode, "config": config.to_dict(), "questions": rows, "latencies": latencies}


def recall(rows, key: str) -> float | None:
    vals = [sum(it[key] for it in r["items"]) / len(r["items"]) for r in rows if r["items"]]
    return sum(vals) / len(vals) if vals else None


def complete(rows, key: str) -> str:
    rows = [r for r in rows if r["items"]]
    return f"{sum(all(it[key] for it in r['items']) for r in rows)}/{len(rows)}"


def by_kind(rows, key: str) -> dict[str, str]:
    out = {}
    for kind in ("text", "heading", "table", "figure", "mixed"):
        its = [it for r in rows for it in r["items"] if it["kind"] == kind]
        if its:
            out[kind] = f"{sum(it[key] for it in its)}/{len(its)}"
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def build_timings(embedder, data_paths: list[str], extracted_dir: Path) -> dict:
    """Cold build, warm re-sync, incremental add, change and removal in a throw-away index directory."""
    tmp = Path(tempfile.mkdtemp(prefix="ew_index_eval_"))
    out = {}
    try:
        t = time.perf_counter()
        r = sync_index(data_paths, embedder, tmp / "cold", extracted_dir)
        out["cold"] = {"seconds": time.perf_counter() - t, "units": r.units_total, "embedded": r.units_embedded,
                       "docs": len(r.added), "embed_seconds": r.timings.get("embed")}
        t = time.perf_counter()
        r = sync_index(data_paths, embedder, tmp / "cold", extracted_dir)
        out["warm"] = {"seconds": time.perf_counter() - t, "reused": len(r.reused), "embedded": r.units_embedded}

        # Incremental: index everything except the largest document, then add it.
        from eventwise.extract import collect_files

        files = collect_files(data_paths)
        largest = max(files, key=lambda p: p.stat().st_size)
        others = [str(p) for p in files if p != largest]
        sync_index(others, embedder, tmp / "incr", extracted_dir)
        t = time.perf_counter()
        r = sync_index([*others, str(largest)], embedder, tmp / "incr", extracted_dir)
        out["incremental_add"] = {"seconds": time.perf_counter() - t, "added": r.added, "reused": len(r.reused),
                                  "embedded": r.units_embedded}
        t = time.perf_counter()
        r = sync_index(others, embedder, tmp / "incr", extracted_dir, prune=True)
        out["prune_remove"] = {"seconds": time.perf_counter() - t, "removed": r.removed, "embedded": r.units_embedded}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(ROOT / ".cache/index"))
    parser.add_argument("--gold", default=str(ROOT / "eval/retrieval_gold.json"))
    parser.add_argument("--out", default=str(ROOT / "eval/results"))
    parser.add_argument("--data", nargs="+", default=["data"])
    parser.add_argument("--skip-build-timing", action="store_true")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    embedder = SentenceTransformerEmbedder(allow_download=False)
    t = time.perf_counter()
    index = IndexStore(args.index).load()
    retriever = HybridRetriever(index, embedder)
    embedder.encode(["warm-up"])
    startup = {"total_s": time.perf_counter() - t, "bm25_build_s": retriever.build_seconds,
               "model_load_s": embedder.load_seconds, "units": len(index.units), "docs": len(index.documents)}
    resolved = resolve_locators(gold, index)

    base = RetrievalConfig()
    plain = replace(base, guaranteed_per_method=0, same_document_decay=1.0, heading_weight=1.0)
    variants = [
        ("hybrid (default)", "hybrid", base),
        ("BM25 only", "bm25", base),
        ("dense only", "dense", base),
        ("plain RRF (no guarantee, no decay, headings weight 1)", "hybrid", plain),
        ("default without per-method guarantee", "hybrid", replace(base, guaranteed_per_method=0)),
        ("default without same-document decay", "hybrid", replace(base, same_document_decay=1.0)),
        ("default plus heading weight 0.5", "hybrid", replace(base, heading_weight=0.5)),
        ("sensitivity: decay 0.7", "hybrid", replace(base, same_document_decay=0.7)),
        ("sensitivity: RRF k=20", "hybrid", replace(base, rrf_k=20)),
        ("sensitivity: hard cap 3 per document", "hybrid", replace(base, max_per_document=3)),
    ]
    runs = [evaluate(retriever, gold, resolved, mode, cfg, label) for label, mode, cfg in variants]
    # Latency: repeat the hybrid run so the numbers are warm and stable.
    lat = []
    for _ in range(5):
        lat += evaluate(retriever, gold, resolved, "hybrid", base, "latency")["latencies"]
    lat.sort()
    latency = {"n": len(lat), "p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * len(lat)) - 1],
               "max_ms": lat[-1]}
    builds = None if args.skip_build_timing else build_timings(embedder, args.data, index.extracted_dir)

    # ------------------------------------------------------------ report
    lines = [f"# Retrieval evaluation — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Index: {startup['docs']} documents, {startup['units']} units, embedder `{index.embedder_name}`. "
             "Gold evidence: `eval/retrieval_gold.json` (scoring only).", "",
             "Recall@k = mean over questions of (required evidence items found in the top k ÷ items required). "
             "`+ctx` also counts items that appear as attached neighbour context of a top-k result.", ""]
    lines += ["## Summary", "", "| Run | Set | Recall@5 | Recall@10 | Recall@10 +ctx | All items @10 | All items @10 +ctx |",
              "|---|---|---|---|---|---|---|"]
    for run in runs:
        for set_name, pred in (("priority (8)", lambda r: r["priority"]),
                               ("development (16)", lambda r: not r["negative"] and not r["heldout"]),
                               ("held-out (10)", lambda r: r["heldout"])):
            rows = [r for r in run["questions"] if pred(r)]
            lines.append(f"| {run['label']} | {set_name} | {pct(recall(rows, 'hit@5'))} | {pct(recall(rows, 'hit@10'))} "
                         f"| {pct(recall(rows, 'ctx@10'))} | {complete(rows, 'hit@10')} | {complete(rows, 'ctx@10')} |")
    hybrid = runs[0]["questions"]
    positives = [r for r in hybrid if not r["negative"]]
    lines += ["", "Development questions were used while designing the ranking; held-out questions (H1–H10) were "
              "written with their gold evidence fixed before they were first run. One default (heading weight 0.5 "
              "-> 1.0) was changed after the first held-out run, so the held-out set is no longer strictly unseen.", ""]
    lines += ["", "## By evidence type (hybrid, development + held-out; items found / items required)", "",
              "| Type | @5 | @10 | @10 +ctx |", "|---|---|---|---|"]
    k5, k10, c10 = by_kind(positives, "hit@5"), by_kind(positives, "hit@10"), by_kind(positives, "ctx@10")
    for kind in k10:
        lines.append(f"| {kind} | {k5[kind]} | {k10[kind]} | {c10[kind]} |")
    lines += ["", "## Per question (hybrid)", "", "| Q | Evidence item | Type | Rank (hybrid) | Rank (BM25) | Rank (dense) | In context @10 |",
              "|---|---|---|---|---|---|---|"]
    others = {run["mode"]: {r["id"]: r for r in run["questions"]} for run in runs[1:3]}
    for r in positives:
        for n, it in enumerate(r["items"]):
            b = others["bm25"][r["id"]]["items"][n]["rank"]
            d = others["dense"][r["id"]]["items"][n]["rank"]
            lines.append(f"| {r['id']}{'*' if r['priority'] else ''} | {it['name']} | {it['kind']} | {it['rank'] or '>10'} "
                         f"| {b or '>10'} | {d or '>10'} | {'yes' if it['ctx@10'] else 'no'} |")
    lines += ["", "`*` = priority question; H = held-out.", "", "## Negative questions (no evidence exists)", "",
              "| Q | Top result | Top fused score | Max dense cosine in top 10 |", "|---|---|---|---|"]
    for r in hybrid:
        if r["negative"]:
            lines.append(f"| {r['id']} | {r['top']} | {r['top_score']:.4f} | {r['top_dense']:.3f} |")
    pos_dense = sorted(r["top_dense"] for r in positives if r["top_dense"] is not None)
    lines.append(f"\nFor comparison, max dense cosine on positive questions: min {pos_dense[0]:.3f}, "
                 f"median {statistics.median(pos_dense):.3f}.")
    lines += ["", "## Latency and start-up", "",
              f"- Start-up (load index, build BM25, load model, warm-up): {startup['total_s']:.2f} s "
              f"(BM25 build {startup['bm25_build_s']:.2f} s, model load {startup['model_load_s'] or 0:.2f} s).",
              f"- Hybrid query latency over {latency['n']} warm queries: p50 {latency['p50_ms']:.1f} ms, "
              f"p95 {latency['p95_ms']:.1f} ms, max {latency['max_ms']:.1f} ms (includes query embedding)."]
    if builds:
        c, w, a, p = builds["cold"], builds["warm"], builds["incremental_add"], builds["prune_remove"]
        lines += ["", "## Index creation and cache reuse (fresh temporary index; extraction cache reused)", "",
                  f"- Cold build: {c['docs']} documents, {c['units']} units embedded in {c['seconds']:.1f} s "
                  f"(embedding {c['embed_seconds']:.1f} s).",
                  f"- Warm re-sync: {w['reused']} documents reused, {w['embedded']} units embedded, {w['seconds']:.2f} s.",
                  f"- Incremental add of {', '.join(a['added'])}: {a['embedded']} units embedded, "
                  f"{a['reused']} documents reused, {a['seconds']:.1f} s.",
                  f"- Prune (remove {', '.join(p['removed'])}): {p['embedded']} units embedded, {p['seconds']:.2f} s."]
    report = "\n".join(lines) + "\n"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.md").write_text(report)
    (out / "retrieval_eval.json").write_text(json.dumps(
        {"startup": startup, "latency": latency, "builds": builds,
         "runs": [{k: v for k, v in run.items() if k != "latencies"} for run in runs]}, indent=1, default=str))
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
