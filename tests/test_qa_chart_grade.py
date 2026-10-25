"""Offline checks for C1 and C4 grading. No API calls."""

import importlib.util
import json
from pathlib import Path

import pytest

GOLD = Path(__file__).resolve().parent.parent / "eval" / "qa_gold.json"
RUNNER = Path(__file__).resolve().parent.parent / "eval" / "run_qa_eval.py"
TEMPERATURE = "d0eb5b5d4e:p3:fig2"
SYNTH = "fff5ac1df9:p26:fig1"
C1_EXPECTED = "22 °C on Friday. It rose every day: Thursday 19, Friday 22, Saturday 26, Sunday 29 °C."
C4_EXPECTED = "SynthLLM-8B (7.4M) scored highest on MATH (71.3); its average was 54.9."
C1_ANSWER = (
    "The maximum daytime temperature on Friday was 22°C. Over the festival, the maximum "
    "daytime temperature increased each day: 19°C on Thursday, 22°C on Friday, 26°C on "
    "Saturday, and 29°C on Sunday."
)
C4_ANSWER = (
    "The highest MATH score is 71.3, achieved by SynthLLM-8B (7.4M). "
    "That model's average is 54.9."
)


def _runner():
    spec = importlib.util.spec_from_file_location("run_qa_eval", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question(question_id):
    if not GOLD.exists():
        pytest.skip("qa_gold.json is not available")
    gold = json.loads(GOLD.read_text())
    return next(q for q in gold["questions"] if q["id"] == question_id)


class _Elements:
    def get_element(self, element_id):
        if element_id == TEMPERATURE:
            return {"caption": "Figure 3: Maximum daytime temperature on site.", "text": "Figure 3"}
        if element_id == "sunday-text":
            return {"text": "Sunday saw the hottest conditions (29°C)."}
        return {"text": "unrelated caption"}


def _capture(answer, element_id, *, images=None, cited_text_id=None):
    cited = cited_text_id or element_id
    return {
        "allowed_ids": [element_id, "sunday-text"],
        "images": list(images) if images is not None else [element_id],
        "citations": [{"element_id": cited, "citation": "source", "kind": "figure"}],
        "raw_citation_ids": [cited],
        "result_ids": [element_id],
        "answer": answer,
        "sufficient": True,
        "error": None,
    }


def test_c1_and_c4_expected_answers_are_unchanged():
    assert _question("C1")["expected_answer"] == C1_EXPECTED
    assert _question("C4")["expected_answer"] == C4_EXPECTED


def test_c1_figure_value_is_supported_when_the_gold_image_is_cited():
    grade = _runner().grade
    graded = grade(_question("C1"), _capture(C1_ANSWER, TEMPERATURE), _Elements(), {}, {}, dry=False)
    sunday = next(item for item in graded["checks"] if item["name"] == "Sunday 29 °C")
    assert sunday["passed"] is True
    assert sunday["supported"] is True
    assert graded["verdict"] == "pass"
    assert graded["unsupported_facts"] == []


def test_c1_text_citation_can_still_support_29_without_an_image():
    runner = _runner()
    question = _question("C1")
    check = next(item for item in question["checks"] if item["name"] == "Sunday 29 °C")
    assert runner.supported(check, question, {"sunday-text"}, set(), "Sunday saw the hottest conditions (29°C).") is True


def test_c1_rejects_a_wrong_temperature_and_a_different_figure():
    grade = _runner().grade
    question = _question("C1")
    wrong_value = C1_ANSWER.replace("29°C", "99°C")
    graded = grade(question, _capture(wrong_value, TEMPERATURE), _Elements(), {}, {}, dry=False)
    assert graded["verdict"] != "pass"

    other = grade(
        question,
        _capture(C1_ANSWER, TEMPERATURE, images=["other-fig"], cited_text_id="other-fig"),
        _Elements(), {}, {}, dry=False,
    )
    assert other["verdict"] != "pass"
    sunday = next(item for item in other["checks"] if item["name"] == "Sunday 29 °C")
    assert sunday["supported"] is False


def test_c1_missing_image_does_not_pass():
    grade = _runner().grade
    graded = grade(
        _question("C1"),
        _capture(C1_ANSWER, TEMPERATURE, images=[]),
        _Elements(), {}, {}, dry=False,
    )
    assert graded["verdict"] != "pass"


def test_c4_accepts_either_word_order_for_the_top_model():
    runner = _runner()
    question = _question("C4")
    for answer in (C4_ANSWER, C4_EXPECTED):
        graded = runner.grade(question, _capture(answer, SYNTH), _Elements(), {}, {}, dry=False)
        assert graded["verdict"] == "pass"
        assert all(item["passed"] for item in graded["checks"])


def test_c4_rejects_a_wrong_model_a_wrong_score_and_a_missing_figure():
    grade = _runner().grade
    question = _question("C4")
    wrong_model = "GPT-4 scored the highest MATH score of 71.3. SynthLLM-8B (7.4M)'s average is 54.9."
    assert grade(question, _capture(wrong_model, SYNTH), _Elements(), {}, {}, dry=False)["verdict"] != "pass"

    wrong_score = "SynthLLM-8B (7.4M) scored highest on MATH (80); its average was 54.9."
    assert grade(question, _capture(wrong_score, SYNTH), _Elements(), {}, {}, dry=False)["verdict"] != "pass"

    missing = grade(question, _capture(C4_ANSWER, SYNTH, images=[]), _Elements(), {}, {}, dry=False)
    assert missing["verdict"] != "pass"
    assert missing["vision_status"] == "retrieved_image_not_attached"
