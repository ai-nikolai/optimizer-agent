"""Lightweight, GPU-free unit checks for the (now fully generic) optimizer agent.

These exercise the generic prompt builder, the rule-based feedback strategy,
and the evaluation parsing. They never touch torch / CUDA / anything problem
specific, so they run anywhere Python 3.9+ is installed.

Run with either ``pytest`` or ``python -m optimizer_agent.tests.test_units``.
"""

from __future__ import annotations

from optimizer_agent.prompts import (
    PromptBuilder,
    GenericFeedbackStrategy,
    NoFeedback,
    get_feedback_strategy,
    FEEDBACK_STRATEGIES,
    classify_result,
    compute_score,
)
from optimizer_agent.evaluation import parse_evaluation, EvalResult


# ----------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------
def test_prompt_builder_fills_present_and_drops_missing():
    cfg = {
        "system": {"order": ["system"], "prompts": {"system": "You are {who}."}},
        "user": {"order": ["task", "impl", "feedback"], "prompts": {
            "task": "## Task",
            "impl": "## Implementation: {current_implementation}",
            "feedback": "## Feedback: {feedback}",
        }},
    }
    builder = PromptBuilder(cfg)
    messages = builder.build_messages({"current_implementation": "def f(): pass", "who": "an expert"})
    roles = {m["role"]: m["content"] for m in messages}
    assert roles["system"] == "You are an expert."
    assert "def f(): pass" in roles["user"]
    # The {feedback} slot was missing -> the whole slot is dropped.
    assert "## Feedback:" not in roles["user"]


def test_default_config_builds_system_and_user():
    from optimizer_agent.prompts import DEFAULT_PROMPT_CONFIG

    builder = PromptBuilder(DEFAULT_PROMPT_CONFIG)
    messages = builder.build_messages({"current_implementation": "x", "feedback": "improve it"})
    roles = {m["role"]: m["content"] for m in messages}
    assert "system" in roles
    assert "x" in roles["user"]
    assert "improve it" in roles["user"]


def test_default_config_drops_missing_feedback_slot():
    from optimizer_agent.prompts import DEFAULT_PROMPT_CONFIG

    builder = PromptBuilder(DEFAULT_PROMPT_CONFIG)
    messages = builder.build_messages({"current_implementation": "x"})
    roles = {m["role"]: m["content"] for m in messages}
    assert "## Feedback:" not in roles["user"]


# ----------------------------------------------------------------------
# classify_result (score-independent)
# ----------------------------------------------------------------------
def test_classify_empty():
    assert classify_result({}) == "empty"


def test_classify_eval_error():
    assert classify_result({"status": "eval_error"}) == "eval_error"


def test_classify_timeout():
    assert classify_result({"timeout": True}) == "eval_error"


def test_classify_incorrect():
    assert classify_result({"correctness": False}) == "incorrect"


def test_classify_incorrect_explicit_false():
    assert classify_result({"correctness": False, "combined_score": 5}) == "incorrect"


def test_classify_improve():
    # A generic signal we don't recognise still falls through to "improve".
    assert classify_result({"combined_score": 3}) == "improve"


def test_classify_improve_default_correctness_true():
    assert classify_result({"compiled": False, "correctness": True}) == "improve"


# ----------------------------------------------------------------------
# compute_score
# ----------------------------------------------------------------------
def test_compute_score_combined():
    assert compute_score({"combined_score": 3.5}) == 3.5


def test_compute_score_none():
    assert compute_score({"compiled": True}) is None


def test_compute_score_empty():
    assert compute_score({}) is None


# ----------------------------------------------------------------------
# Feedback strategies
# ----------------------------------------------------------------------
def test_no_feedback_returns_empty():
    out = NoFeedback().feed_back({"previous_generation": "x"}, {"combined_score": 0.5})
    assert out == {}


def test_generic_feedback_eval_error_message():
    result = {"status": "eval_error", "error": "boom"}
    out = GenericFeedbackStrategy().feed_back({"previous_generation": "prev"}, result)
    assert out["error_case"] == "eval_error"
    assert "boom" in out["feedback"]


def test_generic_feedback_incorrect():
    result = {"correctness": False, "error": "shape mismatch"}
    out = GenericFeedbackStrategy().feed_back({"previous_generation": "prev"}, result)
    assert out["error_case"] == "incorrect"
    assert "shape mismatch" in out["feedback"]


def test_generic_feedback_improve_with_score():
    result = {"correctness": True, "combined_score": 5.0}
    out = GenericFeedbackStrategy().feed_back(
        {"previous_generation": "my code", "prev_score": 3.0}, result
    )
    assert out["error_case"] == "improve"
    assert out["score"] == 5.0
    assert out["prev_score"] == 3.0


def test_generic_feedback_empty_result():
    assert GenericFeedbackStrategy().feed_back({"previous_generation": "x"}, {}) == {}


def test_registry_produces_expected_types():
    assert isinstance(get_feedback_strategy("none"), NoFeedback)
    assert isinstance(get_feedback_strategy("v2"), GenericFeedbackStrategy)
    try:
        get_feedback_strategy("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy")


def test_registry_keys():
    assert set(FEEDBACK_STRATEGIES) == {"none", "v2"}


# ----------------------------------------------------------------------
# evaluation parsing
# ----------------------------------------------------------------------
def test_parse_evaluation_extracts_json():
    stdout = "some log line\n=== Evaluation Result ===\n{\"combined_score\": 4.2, \"correctness\": true}\n"
    res = parse_evaluation(stdout)
    assert res.status == "ok"
    assert res.result["combined_score"] == 4.2
    assert res.result["correctness"] is True


def test_parse_evaluation_missing_marker_yields_fail():
    res = parse_evaluation("no marker here")
    assert res.status == "eval_error"
    assert res.score == float("-inf")


def test_parse_evaluation_nonzero_returncode_is_eval_error():
    res = parse_evaluation("", returncode=1, stderr="kaboom")
    assert res.status == "eval_error"
    assert "kaboom" in res.stderr


def test_eval_result_score_from_combined():
    res = parse_evaluation("=== Evaluation Result ===\n{\"combined_score\": 2.5}")
    assert res.score == 2.5


def test_score_of_helper():
    from optimizer_agent.utils import score_of

    assert score_of({"combined_score": 1.0}) == 1.0
    assert score_of({}) == float("-inf")


def test_extract_final_code_response_pulls_fenced_block():
    from optimizer_agent.utils import extract_final_code_response

    response = "some text\n```python\ndef f(): return 42\n```\nmore text"
    assert extract_final_code_response(response) == "def f(): return 42"


if __name__ == "__main__":
    import traceback

    names = [n for n in dir() if n.startswith("test_")]
    passed = failed = 0
    for name in names:
        fn = globals()[name]
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{len(names)} passed, {failed} failed")
    raise SystemExit(0 if failed == 0 else 1)
