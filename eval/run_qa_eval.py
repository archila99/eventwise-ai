"""End-to-end QA evaluation over the real retrieval + answering pipeline.

    python eval/run_qa_eval.py --dry-run            # retrieval, image selection, prompt size; no API call
    python eval/run_qa_eval.py                      # one OpenAI request per question, no retries
    python eval/run_qa_eval.py --only C1 C4
    python eval/run_qa_eval.py --rescore eval/results/qa_eval_<ts>.json [--manual review.json]

Questions, expected answers and accepted evidence come from eval/qa_gold.json
(gitignored). The pipeline is called exactly as `python -m eventwise.ask` calls
it; the retriever and client are wrapped only to record the search response,
the raw model JSON and token usage.

Automatic verdicts are pass / fail / review. "review" means the answer could not
be judged definitively (partial, or a forbidden pattern matched) and must be
checked by hand; it never counts as correct. A --manual file maps question IDs
to {"final_verdict": "pass|fail", "final_citation_score": ..., "manual_note": "..."}.

Writes eval/results/qa_eval_<timestamp>[_dry].json and .csv (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eventwise import answer as answer_mod  # noqa: E402
from eventwise.answer import allowed_citations, answer_question, parse_model_json  # noqa: E402
from eventwise.config import DEFAULT_INDEX_DIR  # noqa: E402
from eventwise.indexing.index import IndexStore  # noqa: E402
from eventwise.retrieval.retrieve import HybridRetriever  # noqa: E402

TOP_K = 10
GOLD = ROOT / "eval" / "qa_gold.json"
OUT = ROOT / "eval" / "results"
_NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")
_DRY_REPLY = '{"answer": "(dry run)", "citations": [], "sufficient": false}'


# ------------------------------------------------------------------ recording
class RecordingRetriever:
    """Delegates to HybridRetriever and keeps the last search response."""

    def __init__(self, inner: HybridRetriever):
        self._inner = inner
        self.last = None

    def search(self, *args, **kwargs):
        self.last = self._inner.search(*args, **kwargs)
        return self.last

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RecordingClient:
    """Exposes chat.completions.create; records request size, raw reply, usage and latency."""

    def __init__(self, inner=None):
        self._inner = inner
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reset()

    def reset(self):
        self.calls = 0
        self.request = None
        self.raw = None
        self.usage = None
        self.latency_s = None

    def _create(self, **kwargs):
        self.calls += 1
        content = kwargs["messages"][-1]["content"]
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        self.request = {
            "model": kwargs.get("model"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "prompt_chars": sum(len(p.get("text", "")) for p in parts) + len(kwargs["messages"][0]["content"]),
            "images": sum(1 for p in parts if p.get("type") == "image_url"),
        }
        if self._inner is None:
            message = SimpleNamespace(content=_DRY_REPLY)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        start = time.perf_counter()
        completion = self._inner.chat.completions.create(**kwargs)
        self.latency_s = round(time.perf_counter() - start, 2)
        self.raw = completion.choices[0].message.content
        usage = getattr(completion, "usage", None)
        self.usage = usage.model_dump() if hasattr(usage, "model_dump") else None
        return completion


# ------------------------------------------------------------------ capture
def run_question(q: dict, retriever: RecordingRetriever, client: RecordingClient) -> dict:
    retriever.last = None
    client.reset()
    captured: dict = {"error": None}
    start = time.perf_counter()
    try:
        result = answer_question(retriever, q["question"], client, top_k=TOP_K,
                                 documents=q.get("documents"), vision=True)
    except Exception as exc:  # recorded, never retried
        result = None
        captured["error"] = f"{type(exc).__name__}: {exc}"
    captured["total_s"] = round(time.perf_counter() - start, 2)
    resp = retriever.last
    allowed = allowed_citations(resp, retriever) if resp is not None else {}
    data = parse_model_json(client.raw) if isinstance(client.raw, str) else None
    raw_ids = []
    if data and isinstance(data.get("citations"), list):
        raw_ids = [str(c).strip().strip("[]").strip() for c in data["citations"]]
    captured.update({
        "api_calls": client.calls,
        "request": client.request,
        "latency_s": client.latency_s,
        "usage": client.usage,
        "raw_reply": client.raw,
        "raw_citation_ids": raw_ids,
        "result_ids": [r.element_id for r in resp.results] if resp else [],
        "allowed_ids": list(allowed),
        "answer": result.answer if result else None,
        "sufficient": result.sufficient if result else None,
        "citations": [{"element_id": c.element_id, "citation": c.citation, "kind": c.kind}
                      for c in result.citations] if result else [],
        "images": list(result.images) if result else [],
        "warnings": list(result.warnings) if result else [],
        "model": result.model if result else None,
    })
    return captured


# ------------------------------------------------------------------ grading
def numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text or ""):
        try:
            out.append(float(m.rstrip(",").replace(",", "")))
        except ValueError:
            pass
    return out


def num_match(found: list[float], values: list[float], tol: float) -> bool:
    return any(abs(n - v) <= tol + 1e-9 for n in found for v in values)


def check_values(check: dict) -> list[float]:
    return check.get("values") or [check["value"]]


def passes(check: dict, text: str) -> bool:
    if check["type"] == "number":
        return num_match(numbers(text), check_values(check), check.get("tol", 0))
    return re.search(check["regex"], text or "", re.IGNORECASE) is not None


def element_text(retriever, element_id: str, cache: dict) -> str:
    if element_id not in cache:
        try:
            e = retriever.get_element(element_id)
        except KeyError:
            cache[element_id] = ""
            return ""
        table = e.get("table") or {}
        rows = [" | ".join(c or "" for c in row) for row in table.get("raw_cells") or []]
        figure = e.get("figure") or {}
        cache[element_id] = "\n".join(filter(None, [e.get("caption"), e.get("text"), figure.get("text_labels"), *rows]))
    return cache[element_id]


def _gold_figure_shown(q: dict, cited: set[str], images: set[str]) -> bool:
    """The verified figure was cited and its image was attached."""
    return any(e in cited and e in images for e in q.get("visual_elements") or [])


def _cited_text_supports(check: dict, cited_text: str) -> bool:
    if check["type"] == "number":
        return num_match(numbers(cited_text), check_values(check), check.get("tol", 0))
    return re.search(check.get("support_regex", check["regex"]), cited_text, re.IGNORECASE) is not None


def supported(check: dict, q: dict, cited: set[str], images: set[str], cited_text: str) -> bool | None:
    mode = check.get("support", "text")
    if mode == "none":
        return None
    if mode == "visual":
        return _gold_figure_shown(q, cited, images)
    if mode == "derived":
        return all(num_match(numbers(cited_text), [v], 0) for v in check["inputs"])
    if mode == "text_or_visual":
        # The required value may live only in the verified figure. Another image,
        # or a different number, does not satisfy the check.
        return _cited_text_supports(check, cited_text) or _gold_figure_shown(q, cited, images)
    if check["type"] == "number":
        return num_match(numbers(cited_text), check_values(check), check.get("tol", 0))
    return re.search(check.get("support_regex", check["regex"]), cited_text, re.IGNORECASE) is not None


def doc_prefix(element_id: str) -> str:
    return element_id.split(":", 1)[0]


def grade(q: dict, cap: dict, retriever, defects: dict, cache: dict, dry: bool) -> dict:
    allowed = set(cap["allowed_ids"])
    images = set(cap["images"])
    valid = [c["element_id"] for c in cap["citations"]]
    cited = set(valid)
    invalid = [i for i in cap["raw_citation_ids"] if i and i not in allowed]
    g: dict = {}

    # Retrieval and citation coverage per evidence group.
    groups = []
    for group in q.get("evidence", []):
        ids = set(group["any_of"])
        docs = {doc_prefix(i) for i in ids}
        groups.append({
            "name": group["name"], "optional": bool(group.get("optional")),
            "retrieved": bool(ids & allowed), "in_results": bool(ids & set(cap["result_ids"])),
            "cited": bool(ids & cited), "cited_same_doc": any(doc_prefix(c) in docs for c in cited),
        })
    required = [x for x in groups if not x["optional"]]
    relevant = set().union(*(set(x["any_of"]) for x in q.get("evidence", []))) if q.get("evidence") else set()
    irrelevant = [c for c in valid if c not in relevant]
    g["evidence_groups"] = groups
    g["retrieval_ok"] = all(x["retrieved"] for x in required)
    g["missing_retrieval"] = [x["name"] for x in required if not x["retrieved"]]
    g["label_defects_in_prompt"] = [e for e in allowed if e in defects]
    g["label_defects_cited"] = [{**defects[c["element_id"]], "rendered_citation": c["citation"]}
                                for c in cap["citations"] if c["element_id"] in defects]

    visual = []
    for e in q.get("visual_elements", []):
        visual.append({"element_id": e, "retrieved": e in allowed, "in_results": e in cap["result_ids"],
                       "image_attached": e in images, "cited": e in cited})
    g["visual"] = visual
    if not visual:
        g["vision_status"] = "n/a"
    elif not all(v["retrieved"] for v in visual):
        g["vision_status"] = "figure_not_retrieved"
    elif not all(v["image_attached"] for v in visual):
        g["vision_status"] = "retrieved_image_not_attached"
    else:
        g["vision_status"] = "image_attached"

    if dry or cap["error"]:
        g.update({"verdict": "error" if cap["error"] else None, "reason": cap["error"],
                  "failure_type": "execution_error" if cap["error"] else None})
        return g

    answer = cap["answer"] or ""
    cited_text = "\n".join(element_text(retriever, c, cache) for c in valid)

    # Citation scoring.
    g["invalid_citations"] = invalid
    g["irrelevant_citations"] = irrelevant
    g["all_required_sources_cited"] = all(x["cited"] for x in required) if required else None
    g["missing_sources"] = [x["name"] for x in required if not x["cited"]]
    if not required:
        g["citation_score"] = None
    elif all(x["cited"] for x in required):
        g["citation_score"] = 1.0 if not irrelevant and not invalid else 0.5
    elif any(x["cited"] or x["cited_same_doc"] for x in required):
        g["citation_score"] = 0.5
    else:
        g["citation_score"] = 0.0

    if not q["answerable"]:
        extra = [n for n in numbers(answer) if n not in q.get("allowed_numbers", []) and n not in numbers(q["question"])]
        g["unexpected_numbers"] = extra
        if cap["sufficient"]:
            g.update(verdict="fail", reason="missed abstention (sufficient=true)", failure_type="generation_failure")
        elif extra:
            g.update(verdict="review", reason=f"abstained but states other numbers {extra}", failure_type=None)
        else:
            g.update(verdict="pass", reason="correct abstention", failure_type=None)
        return g

    results = []
    for check in q["checks"]:
        ok = passes(check, answer)
        results.append({"name": check["name"], "passed": ok, "support": check.get("support", "text"),
                        "supported": supported(check, q, cited, images, cited_text) if ok else None})
    forbidden = [f["name"] for f in q.get("forbidden", []) if passes(f, answer)]
    g["checks"] = results
    g["forbidden_hits"] = forbidden
    g["unsupported_facts"] = [r["name"] for r in results if r["passed"] and r["supported"] is False]
    n_pass = sum(r["passed"] for r in results)
    if not cap["sufficient"]:
        g.update(verdict="fail", reason="false abstention (sufficient=false)")
    elif n_pass == len(results) and not forbidden and g["unsupported_facts"]:
        g.update(verdict="review", reason=f"all facts present but not supported by citations: {g['unsupported_facts']}")
    elif n_pass == len(results) and not forbidden:
        g.update(verdict="pass", reason="all required facts present")
    elif n_pass == 0:
        g.update(verdict="fail", reason="no required fact present" + (f"; forbidden: {forbidden}" if forbidden else ""))
    else:
        g.update(verdict="review", reason=f"{n_pass}/{len(results)} facts" + (f"; forbidden: {forbidden}" if forbidden else ""))

    visual_checks = [r for r in results if r["support"] == "visual"]
    if visual_checks:
        if g["vision_status"] == "figure_not_retrieved":
            g["visual_diagnosis"] = "retrieval_failure"
        elif all(r["passed"] for r in visual_checks):
            g["visual_diagnosis"] = "ok" if g["vision_status"] == "image_attached" else "correct_without_image"
        elif g["vision_status"] == "image_attached":
            g["visual_diagnosis"] = "visual_interpretation_failure"
        else:
            g["visual_diagnosis"] = "vision_selection_failure"
    else:
        g["visual_diagnosis"] = "n/a"

    if g["verdict"] == "pass":
        g["failure_type"] = None
    elif not g["retrieval_ok"]:
        g["failure_type"] = "retrieval_failure"
    elif g["visual_diagnosis"] in ("vision_selection_failure", "visual_interpretation_failure"):
        g["failure_type"] = g["visual_diagnosis"]
    else:
        g["failure_type"] = "generation_failure"
    return g


# ------------------------------------------------------------------ report
def summarise(records: list[dict], dry: bool, prices: tuple[float | None, float | None]) -> dict:
    s: dict = {"questions": len(records)}
    s["retrieval_ok"] = sum(r["grading"]["retrieval_ok"] for r in records if r["answerable"])
    s["answerable"] = sum(r["answerable"] for r in records)
    s["vision_status"] = dict(Counter(r["grading"]["vision_status"] for r in records if r["grading"]["vision_status"] != "n/a"))
    s["images_sent_total"] = sum(len(r["captured"]["images"]) for r in records)
    s["prompt_chars_total"] = sum((r["captured"]["request"] or {}).get("prompt_chars", 0) for r in records)
    s["model_requests"] = sum(r["captured"]["api_calls"] for r in records)
    s["requests_stubbed"] = dry
    s["errors"] = [r["id"] for r in records if r["captured"]["error"]]
    if dry:
        return s
    final = [r.get("final_verdict") or r["grading"]["verdict"] for r in records]
    s["auto_verdicts"] = dict(Counter(r["grading"]["verdict"] for r in records))
    s["final_verdicts"] = dict(Counter(final))
    s["answer_accuracy"] = round(final.count("pass") / len(records), 3) if records else None
    by_cat = defaultdict(Counter)
    for r, v in zip(records, final):
        by_cat[r["category"]][v] += 1
    s["by_category"] = {k: dict(v) for k, v in by_cat.items()}
    scores = [r.get("final_citation_score", r["grading"].get("citation_score")) for r in records]
    scores = [x for x in scores if x is not None]
    s["citation_score_mean"] = round(sum(scores) / len(scores), 3) if scores else None
    multi = [r for r in records if r["grading"].get("all_required_sources_cited") is not None]
    s["all_required_sources_cited"] = f"{sum(r['grading']['all_required_sources_cited'] for r in multi)}/{len(multi)}"
    s["invalid_citations"] = sum(len(r["grading"].get("invalid_citations") or []) for r in records)
    s["irrelevant_citations"] = sum(len(r["grading"].get("irrelevant_citations") or []) for r in records)
    s["unsupported_facts"] = {r["id"]: r["grading"]["unsupported_facts"] for r in records if r["grading"].get("unsupported_facts")}
    failure = [r.get("failure_type_manual") or r["grading"].get("failure_type") for r in records
               if (r.get("final_verdict") or r["grading"]["verdict"]) != "pass"]
    s["failure_types"] = dict(Counter(f for f in failure if f))
    s["visual_diagnosis"] = {r["id"]: r["grading"].get("visual_diagnosis") for r in records if r["grading"].get("visual_diagnosis") not in (None, "n/a")}
    s["label_defects_cited"] = {r["id"]: r["grading"]["label_defects_cited"] for r in records if r["grading"]["label_defects_cited"]}
    s["label_defects_in_prompt"] = {r["id"]: r["grading"]["label_defects_in_prompt"] for r in records if r["grading"]["label_defects_in_prompt"]}
    s["manual_overrides"] = [r["id"] for r in records if "final_verdict" in r or "final_citation_score" in r]
    usage = [r["captured"]["usage"] or {} for r in records]
    s["tokens"] = {
        "prompt": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion": sum(u.get("completion_tokens") or 0 for u in usage),
        "cached_prompt": sum(((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for u in usage),
    }
    s["api_reported_cost"] = None  # the Chat Completions response carries no cost field
    if prices[0] is not None and prices[1] is not None:
        s["estimated_cost_usd"] = round(s["tokens"]["prompt"] / 1e6 * prices[0] + s["tokens"]["completion"] / 1e6 * prices[1], 4)
    latencies = [r["captured"]["latency_s"] for r in records if r["captured"]["latency_s"] is not None]
    s["latency_s"] = {"mean": round(sum(latencies) / len(latencies), 2), "max": max(latencies)} if latencies else None
    return s


CSV_FIELDS = ["id", "category", "auto_verdict", "final_verdict", "reason", "sufficient", "checks_passed",
              "forbidden_hits", "citation_score", "all_required_sources_cited", "missing_sources",
              "invalid_citations", "irrelevant_citations", "unsupported_facts", "retrieval_ok", "missing_retrieval",
              "vision_status", "visual_diagnosis", "images_attached", "failure_type", "label_defects_cited",
              "prompt_tokens", "completion_tokens", "latency_s", "error", "answer"]


def csv_row(r: dict) -> dict:
    g, c = r["grading"], r["captured"]
    checks = g.get("checks") or []
    join = lambda xs: "; ".join(str(x) for x in xs or [])  # noqa: E731
    return {
        "id": r["id"], "category": r["category"], "auto_verdict": g.get("verdict"),
        "final_verdict": r.get("final_verdict") or g.get("verdict"), "reason": g.get("reason"),
        "sufficient": c["sufficient"], "checks_passed": f"{sum(x['passed'] for x in checks)}/{len(checks)}" if checks else "",
        "forbidden_hits": join(g.get("forbidden_hits")),
        "citation_score": r.get("final_citation_score", g.get("citation_score")),
        "all_required_sources_cited": g.get("all_required_sources_cited"), "missing_sources": join(g.get("missing_sources")),
        "invalid_citations": join(g.get("invalid_citations")), "irrelevant_citations": join(g.get("irrelevant_citations")),
        "unsupported_facts": join(g.get("unsupported_facts")), "retrieval_ok": g["retrieval_ok"],
        "missing_retrieval": join(g["missing_retrieval"]), "vision_status": g["vision_status"],
        "visual_diagnosis": g.get("visual_diagnosis"), "images_attached": join(c["images"]),
        "failure_type": g.get("failure_type"),
        "label_defects_cited": join(d["shown"] + " -> " + d["correct"] for d in g["label_defects_cited"]),
        "prompt_tokens": (c["usage"] or {}).get("prompt_tokens"), "completion_tokens": (c["usage"] or {}).get("completion_tokens"),
        "latency_s": c["latency_s"], "error": c["error"], "answer": c["answer"],
    }


def write(path_stem: Path, report: dict) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    path_stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    with path_stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in report["questions"]:
            writer.writerow(csv_row(r))


def print_report(report: dict) -> None:
    dry = report["run"]["mode"] == "dry"
    for r in report["questions"]:
        g, c = r["grading"], r["captured"]
        verdict = "" if dry else f"{(r.get('final_verdict') or g['verdict'] or ''):7} cite={g.get('citation_score')} "
        print(f"{r['id']:3} {r['category']:15} {verdict}retrieval={'ok' if g['retrieval_ok'] else 'MISS ' + str(g['missing_retrieval'])} "
              f"vision={g['vision_status']} images={c['images']} chars={(c['request'] or {}).get('prompt_chars')}"
              + (f"  [{g.get('reason')}]" if not dry else ""))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end QA evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Retrieval and prompt building only; no API call")
    parser.add_argument("--only", nargs="+", help="Question IDs to run")
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--index", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rescore", help="Re-grade a saved report without calling the API")
    parser.add_argument("--manual", help="JSON of manual verdicts to merge (with --rescore)")
    parser.add_argument("--price-input", type=float, help="USD per 1M prompt tokens, for a cost estimate")
    parser.add_argument("--price-output", type=float, help="USD per 1M completion tokens, for a cost estimate")
    args = parser.parse_args(argv)
    os.chdir(ROOT)

    gold = json.loads(Path(args.gold).read_text())
    defects = {d["element_id"]: d for d in gold.get("known_label_defects", [])}
    questions = [q for q in gold["questions"] if not args.only or q["id"] in args.only]
    index = IndexStore(args.index).load()
    cache: dict = {}
    prices = (args.price_input, args.price_output)

    if args.rescore:
        report = json.loads(Path(args.rescore).read_text())
        manual = json.loads(Path(args.manual).read_text()) if args.manual else {}
        retriever = HybridRetriever(index, None)
        by_id = {q["id"]: q for q in gold["questions"]}
        for r in report["questions"]:
            r["grading"] = grade(by_id[r["id"]], r["captured"], retriever, defects, cache, dry=False)
            r.update(manual.get(r["id"]) or {})
        report["summary"] = summarise(report["questions"], False, prices)
        report["run"]["rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stem = Path(args.rescore).with_suffix("")
        write(stem.with_name(stem.name.removesuffix("_final") + "_final"), report)
        print_report(report)
        return 0

    client = RecordingClient()
    if not args.dry_run:
        from eventwise.ask import _load_local_env
        _load_local_env()
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
            return 2
        from openai import OpenAI
        client = RecordingClient(OpenAI(max_retries=0))

    from eventwise.indexing.embeddings import SentenceTransformerEmbedder
    retriever = RecordingRetriever(HybridRetriever(index, SentenceTransformerEmbedder(allow_download=False)))
    records = []
    for q in questions:
        cap = run_question(q, retriever, client)
        records.append({"id": q["id"], "category": q["category"], "answerable": q["answerable"],
                        "question": q["question"], "captured": cap,
                        "grading": grade(q, cap, retriever, defects, cache, dry=args.dry_run)})
        print(f"{q['id']} done ({cap['total_s']} s){' ERROR ' + cap['error'] if cap['error'] else ''}", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run": {"timestamp": stamp, "mode": "dry" if args.dry_run else "live", "model": answer_mod.DEFAULT_MODEL,
                "reasoning_effort": answer_mod.REASONING_EFFORT, "top_k": TOP_K, "vision": True,
                "max_images": answer_mod.MAX_IMAGES, "client_max_retries": 0, "gold": str(args.gold)},
        "summary": summarise(records, args.dry_run, prices),
        "questions": records,
    }
    write(OUT / f"qa_eval_{stamp}{'_dry' if args.dry_run else ''}", report)
    print_report(report)
    print(f"\nWrote {OUT / f'qa_eval_{stamp}'}{'_dry' if args.dry_run else ''}.json/.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
