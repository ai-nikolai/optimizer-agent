"""Lightweight, GPU-free unit checks for the optimizer agent.

These exercise the single generic prompt builder (``PromptBuilder``) and the
rule-based feedback strategy (``GenericFeedbackStrategy``) from
``optimizer_agent.prompts_v2``, plus the evaluation helpers. They never touch
torch / CUDA, so they run anywhere Python 3.9+ is installed.

Run with either ``pytest`` or ``python -m optimizer_agent.tests.test_units``.
"""

from __future__ import annotations

from optimizer_agent.prompts_v2 import (
    PromptBuilder,
    GenericFeedbackStrategy,
    NoFeedback,
    get_feedback_strategy,
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
    b = PromptBuilder(cfg)
    messages = b.build_messages(
        {"current_implementation": "def f(): pass", "who": "an expert"}
    )
    roles = {m["role"]: m["content"] for m in messages}
    assert roles["system"] == "You are an expert."
    assert "def f(): pass" in roles["user"]
    # The {feedback} slot was missing -> the whole slot is dropped.
    assert "## Feedback:" not in roles["user"]


def test_default_config_builds_system_and_user():
    from optimizer_agent.prompts_v2 import DEFAULT_PROMPT_CONFIG

    b = PromptBuilder(DEFAULT_PROMPT_CONFIG)
    messages = b.build_messages({"current_implementation": "x", "reference": "y"})
    roles = {m["role"]: m["content"] for m in messages}
    assert "system" in roles
    assert "x" in roles["user"]
    assert "y" in roles["user"]


# ----------------------------------------------------------------------
# classify_result (score-independent)
# ----------------------------------------------------------------------
def test_classify_empty():
    assert classify_result({}) == "empty"


def test_classify_compile_error():
    assert classify_result({"compiled": False}) == "compile_error"


def test_classify_runtime_error():
    assert classify_result({"compiled": True, "loading_success": False}) == "runtime_error"


def test_classify_incorrect():
    assert classify_result({"compiled": True, "loading_success": True, "correctness": False}) == "incorrect"


def test_classify_improve():
    assert classify_result({"compiled": True, "loading_success": True, "correctness": True}) == "improve"


def test_classify_cuda_shape_loading_failure():
    assert (
        classify_result({"compiled": True, "loading_success": False, "error": "bad"})
        == "runtime_error"
    )


# ----------------------------------------------------------------------
# compute_score
# ----------------------------------------------------------------------
def test_compute_score_speedup():
    result = {"runtime_custom_ms": 5.0, "runtime_original_ms": 40.0}
    assert compute_score(result) == 8.0


def test_compute_score_additive_combined():
    # evaluate.py already puts original - custom into combined_score.
    result = {"combined_score": 3.5, "runtime_custom_ms": 1.0, "runtime_original_ms": 4.5}
    assert compute_score(result) == 3.5


def test_compute_score_none():
    assert compute_score({"compiled": True, "correctness": True}) is None


# ----------------------------------------------------------------------
# Feedback strategies
# ----------------------------------------------------------------------
def test_no_feedback_returns_empty():
    out = NoFeedback().feed_back({"previous_generation": "x"}, {"combined_score": 0.5})
    assert out == {}


def test_generic_feedback_runtime_error_message():
    result = {"compiled": True, "loading_success": False, "error": "boom"}
    out = GenericFeedbackStrategy().feed_back({"previous_generation": "prev"}, result)
    assert out["error_case"] == "runtime_error"
    assert out["error_message"] == "boom"


def test_generic_feedback_improve_with_speedup():
    result = {"compiled": True, "loading_success": True, "correctness": True,
              "runtime_custom_ms": 2.0, "runtime_original_ms": 10.0}
    out = GenericFeedbackStrategy().feed_back(
        {"previous_generation": "my code", "prev_score": 3.0}, result
    )
    assert out["error_case"] == "improve"
    assert out["score"] == 5.0
    assert out["prev_score"] == 3.0


def test_registry_produces_expected_types():
    assert isinstance(get_feedback_strategy("none"), NoFeedback)
    assert isinstance(get_feedback_strategy("v2"), GenericFeedbackStrategy)
    try:
        get_feedback_strategy("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy")


# ----------------------------------------------------------------------
# evaluation parsing
# ----------------------------------------------------------------------
def test_parse_evaluation_extracts_json():
    stdout = "some log line\n=== Evaluation Result ===\n{\"combined_score\": 4.2, \"correctness\": true}\n"
    res = parse_evaluation(stdout)
    assert res.result["combined_score"] == 4.2
    assert res.result["correctness"] is True


def test_parse_evaluation_missing_marker_yields_fail():
    res = parse_evaluation("no marker here")
    assert res.status != "ok" or res.result.get("combined_score", 0) == float("-inf")


def test_parse_evaluation_nonzero_returncode_is_eval_error():
    res = parse_evaluation("", returncode=1, stderr="kaboom")
    assert res.status == "eval_error"
    assert "kaboom" in res.stderr


def test_eval_result_failure_class_cuda_shape():
    stdout = "=== Evaluation Result ===\n{\"compiled\": false, \"error\": \"x\"}"
    res = parse_evaluation(stdout)
    assert res.compiled is False
    assert res.failure_class == "compile_error"


if __name__ == "__main__":
    import traceback

    names = [n for n in dir() if n.startswith("test_")]
    passed = 0
    failed = 0
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
