"""Offline checks for the T4 grader. No API calls."""

import importlib.util
import json
from pathlib import Path

import pytest

GOLD = Path(__file__).resolve().parent.parent / "eval" / "qa_gold.json"
RUNNER = Path(__file__).resolve().parent.parent / "eval" / "run_qa_eval.py"
TABLE = "fff5ac1df9:p135:tab2"
EXPECTED = "BBQ, Cybench and SimpleQA (only these three)."


def _runner():
    spec = importlib.util.spec_from_file_location("run_qa_eval", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _t4():
    if not GOLD.exists():
        pytest.skip("qa_gold.json is not available")
    gold = json.loads(GOLD.read_text())
    return next(q for q in gold["questions"] if q["id"] == "T4")


class _ElementText:
    def get_element(self, element_id):
        assert element_id == TABLE
        return {"text": "BBQ Cybench SimpleQA"}


def _capture(answer):
    return {
        "allowed_ids": [TABLE],
        "images": [],
        "citations": [{"element_id": TABLE, "citation": "ai_index_report_2026.pdf — p. 135", "kind": "table"}],
        "raw_citation_ids": [TABLE],
        "result_ids": [TABLE],
        "answer": answer,
        "sufficient": True,
        "error": None,
    }


def test_t4_expected_answer_is_unchanged_and_hle_is_unexpected():
    question = _t4()
    assert question["expected_answer"] == EXPECTED
    names = [item["name"] for item in question["forbidden"]]
    assert "Lists HLE, an unexpected benchmark" in names


def test_t4_answer_that_adds_hle_is_not_a_pass():
    grade = _runner().grade
    question = _t4()
    graded = grade(
        question,
        _capture("Claude Opus 4.5 reported results for BBQ, Cybench, SimpleQA, and HLE."),
        _ElementText(), {}, {}, dry=False,
    )
    assert graded["verdict"] != "pass"
    assert graded["forbidden_hits"] == [
        "Lists HLE, an unexpected benchmark",
    ]
    assert graded["checks"] and all(item["passed"] for item in graded["checks"])


def test_t4_answer_with_only_the_three_benchmarks_passes():
    grade = _runner().grade
    question = _t4()
    graded = grade(
        question,
        _capture("Claude Opus 4.5 reported BBQ, Cybench and SimpleQA."),
        _ElementText(), {}, {}, dry=False,
    )
    assert graded["verdict"] == "pass"
    assert graded["forbidden_hits"] == []
    assert graded["citation_score"] == 1.0
