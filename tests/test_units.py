"""Small, GPU-free unit and integration tests.

Run with:  python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from optimizer_agent.candidates import (
    CandidatesBestNAll,
    CandidatesBestNLast,
    MapElitesCandidates,
    SingleCandidate,
)
from optimizer_agent.diffs import (
    DiffApplyError,
    apply_json_edits,
    apply_unified_diff,
)
from optimizer_agent.feedback import (
    COMPILE_ERROR,
    EMPTY,
    EVAL_ERROR,
    IMPROVE,
    INCORRECT,
    INVALID,
    KIND_VALID,
    RUNTIME_ERROR,
    apply_feedback,
    build_feedback_strategy,
    classify_result,
    failure_kind,
    feedback_hint,
    metric_change_of,
    speedup_of,
)
from optimizer_agent.llm import LLMError, LLMServer
from optimizer_agent.main import Agent
from optimizer_agent.prompts import DEFAULT_PROMPT_CONFIG, create_prompt
from optimizer_agent.utils import (
    extract_eval_results,
    extract_final_code_response,
    extract_response,
    match_largest,
    match_last,
    match_last_code,
    rank_score_of,
    run_evaluation,
    score_of,
    summarize_result,
)

EVAL_SCRIPT = """\
import argparse, json, pathlib, sys
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
args = parser.parse_args()
code = pathlib.Path(args.solution).read_text()
if "BROKEN" in code:
    print("intentional failure", file=sys.stderr)
    sys.exit(1)
print("=== Evaluation Result ===")
print(json.dumps({"combined_score": 1.0, "correctness": True}))
"""

CORRECTNESS_EVAL_SCRIPT = """\
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
args = parser.parse_args()
code = pathlib.Path(args.solution).read_text()
if "WRONG" in code:
    result = {"correctness": False, "error": "output mismatch"}
else:
    result = {"correctness": True, "combined_score": 2.0}
print("=== Evaluation Result ===")
print(json.dumps(result))
"""

FEEDBACK_EVAL_SCRIPT = """\
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
parser.parse_args()
print("=== Evaluation Result ===")
print(json.dumps({
    "combined_score": 1.0,
    "valid": True,
    "feedback": "EVALUATOR HINT: make the next attempt shorter",
}))
"""

ONESHOT_FEEDBACK_EVAL_SCRIPT = """\
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
args = parser.parse_args()
code = pathlib.Path(args.solution).read_text()
result = {"combined_score": 1.0, "valid": True}
if "FIRST" in code:
    result["combined_score"] = 0.5
    result["feedback"] = "STALE EVALUATOR HINT"
print("=== Evaluation Result ===")
print(json.dumps(result))
"""

INVALID_STAGE_EVAL_SCRIPT = """\
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
parser.parse_args()
print("=== Evaluation Result ===")
print(json.dumps({
    "combined_score": -1.0,
    "valid": False,
    "failure_stage": "invalid",
    "error": "constraint failed",
}))
"""

INVALID_THEN_VALID_EVAL_SCRIPT = """\
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--solution", required=True)
parser.add_argument("--reference", default=None)
args = parser.parse_args()
code = pathlib.Path(args.solution).read_text()
valid = "FIRST" not in code
print("=== Evaluation Result ===")
print(json.dumps({
    "combined_score": 1.0 if valid else -1.0,
    "valid": valid,
    "failure_stage": "invalid" if not valid else "improve",
    "error": "" if valid else "constraint failed",
}))
"""


class RecordingServer:
    """Deterministic stand-in for the LLM: returns queued fenced code."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call_server(self, messages, **kwargs):
        self.calls.append(messages)
        response = self.responses.pop(0) if self.responses else "```python\nX = 0\n```"
        return [response], []


class PromptTests(unittest.TestCase):
    def test_missing_slots_are_dropped(self):
        messages = create_prompt(
            "core",
            {"current_artifact": "CODE"},
            DEFAULT_PROMPT_CONFIG["improve"],
        )
        user = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("CODE", user)
        self.assertIn("## Task", user)
        self.assertNotIn("## Feedback", user)
        self.assertNotIn("## Result", user)

    def test_present_slots_are_rendered(self):
        messages = create_prompt(
            "core",
            {
                "current_artifact": "CODE",
                "reference": "REF",
                "result": "score=1",
                "feedback": "try harder",
            },
            DEFAULT_PROMPT_CONFIG["improve"],
        )
        user = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("REF", user)
        self.assertIn("score=1", user)
        self.assertIn("try harder", user)

    def test_flat_config_is_supported(self):
        flat = {
            "system": {"order": [], "prompts": {}},
            "user": {
                "order": ["implementation"],
                "prompts": {"implementation": "Code: {current_implementation}"},
            },
        }
        messages = create_prompt("core", {"current_implementation": "X"}, flat)
        self.assertEqual(messages[-1]["content"], "Code: X")


class ParsingTests(unittest.TestCase):
    def test_extract_fenced_code(self):
        text = "preamble\n```python\nprint(1)\n```\ntrailing"
        self.assertEqual(extract_final_code_response(text), "print(1)")

    def test_extract_unfenced_returns_input(self):
        self.assertEqual(extract_final_code_response("print(1)"), "print(1)")

    def test_extract_response_raw_and_json(self):
        text = "Here is the artifact:\n```python\nX = 1\n```\nThanks"
        self.assertEqual(extract_response(text, "code"), "X = 1")
        self.assertEqual(extract_response(text, "raw"), text.strip())
        self.assertEqual(
            extract_response('prefix {"a": 1, "b": [2]} suffix', "json"),
            '{"a": 1, "b": [2]}',
        )

    def test_extract_eval_result(self):
        stdout = 'log\n=== Evaluation Result ===\n{"combined_score": 2.5}\n'
        self.assertEqual(extract_eval_results(stdout)["combined_score"], 2.5)

    def test_missing_marker_is_none(self):
        self.assertIsNone(extract_eval_results("no marker"))

    def test_score_of_handles_bad_values(self):
        self.assertEqual(score_of({"combined_score": 3}), 3.0)
        self.assertEqual(score_of({}), float("-inf"))
        self.assertEqual(score_of({"combined_score": "nope"}), float("-inf"))

    def test_summarize_failure_mentions_status(self):
        result = {
            "status": "eval_error",
            "scores": {"combined_score": float("-inf")},
            "stderr": "boom",
            "stdout": "",
        }
        self.assertIn("eval_error", summarize_result(result))
        self.assertIn("boom", summarize_result(result))


class FeedbackTests(unittest.TestCase):
    @staticmethod
    def _result(scores, status="ok", stderr="", stdout=""):
        return {"status": status, "scores": scores, "stderr": stderr, "stdout": stdout}

    def test_empty_code(self):
        self.assertEqual(classify_result(self._result({}), code=""), EMPTY)

    def test_compile_error_flag(self):
        result = self._result({"compiled": False, "error": "bad source"})
        self.assertEqual(classify_result(result, code="X=1"), COMPILE_ERROR)

    def test_compile_error_from_text(self):
        result = self._result({}, status="eval_error", stderr="nvcc fatal: bad flag")
        self.assertEqual(
            classify_result(result, code="X=1", use_text_heuristics=True),
            COMPILE_ERROR,
        )

    def test_generic_evaluator_crash_is_not_runtime_error(self):
        result = self._result({}, status="eval_error", stderr="Traceback: evaluator bug")
        self.assertEqual(classify_result(result, code="artifact"), EVAL_ERROR)

    def test_runtime_error_flag(self):
        result = self._result({"loading_success": False, "error": "launch failed"})
        self.assertEqual(classify_result(result, code="X=1"), RUNTIME_ERROR)

    def test_incorrect_flag(self):
        result = self._result({"correctness": False, "error": "mismatch"})
        self.assertEqual(classify_result(result, code="X=1"), INCORRECT)

    def test_generic_valid_false(self):
        result = self._result({"valid": False, "error": "constraint violated"})
        self.assertEqual(classify_result(result, code="artifact"), INVALID)

    def test_evaluator_can_classify_itself(self):
        result = self._result(
            {"failure_kind": "invalid", "error_kind": "constraint", "score": 0.2}
        )
        self.assertEqual(classify_result(result, code="artifact"), INVALID)
        self.assertEqual(failure_kind(INVALID), "invalid")

    def test_generic_score_is_valid(self):
        result = self._result({"score": 0.9, "valid": True})
        self.assertEqual(
            classify_result(result, code="artifact", score_key="score"), IMPROVE
        )
        self.assertEqual(failure_kind(IMPROVE), KIND_VALID)

    def test_apply_feedback_custom_score_key(self):
        prompt_data = {}
        result = self._result({"score": 0.9, "valid": True})
        stage = apply_feedback(
            prompt_data, result, code="artifact", score_key="score"
        )
        self.assertEqual(stage, IMPROVE)
        self.assertEqual(prompt_data["failure_kind"], KIND_VALID)
        self.assertEqual(prompt_data["score"], 0.9)
        self.assertEqual(prompt_data["metric_name"], "score")

    def test_custom_hints(self):
        hint = feedback_hint(
            INVALID,
            scores={"error_kind": "constraint"},
            error="bad value",
            profile="generic",
            hints={INVALID: "CUSTOM: {error_kind} - {detail}"},
        )
        self.assertIn("CUSTOM", hint)
        self.assertIn("constraint", hint)
        self.assertIn("bad value", hint)

    def test_kernel_profile_hint(self):
        hint = feedback_hint(
            COMPILE_ERROR, scores={}, error="nvcc fatal", profile="kernel"
        )
        self.assertIn("compile", hint.lower())
        self.assertIn("nvcc fatal", hint)

    def test_eval_error(self):
        result = self._result({}, status="eval_error", stderr="plain crash")
        self.assertEqual(classify_result(result, code="X=1"), EVAL_ERROR)

    def test_improve_and_speedup(self):
        result = self._result(
            {
                "combined_score": 1.0,
                "correctness": True,
                "runtime_custom_ms": 5.0,
                "runtime_original_ms": 10.0,
            }
        )
        self.assertEqual(classify_result(result, code="X=1"), IMPROVE)
        self.assertAlmostEqual(speedup_of(result["scores"]), 2.0)

    def test_metric_change_lower_is_better(self):
        change, improved = metric_change_of(
            {"runtime_custom_ms": 5.0, "runtime_original_ms": 10.0},
            score_key="runtime_custom_ms",
            baseline_key="runtime_original_ms",
            higher_is_better=False,
        )
        self.assertAlmostEqual(change, 2.0)
        self.assertTrue(improved)

    def test_metric_change_higher_is_better(self):
        change, improved = metric_change_of(
            {"combined_score": 120.0, "baseline_score": 100.0},
            score_key="combined_score",
            baseline_key="baseline_score",
            higher_is_better=True,
        )
        self.assertAlmostEqual(change, 1.2)
        self.assertTrue(improved)

    def test_apply_feedback_sets_prompt_keys(self):
        prompt_data = {}
        result = self._result(
            {"correctness": False, "error": "output mismatch"}, status="ok"
        )
        stage = apply_feedback(prompt_data, result, code="BAD = 1")
        self.assertEqual(stage, INCORRECT)
        self.assertEqual(prompt_data["failure_stage"], INCORRECT)
        self.assertIn("incorrect", prompt_data["feedback"].lower())
        self.assertIn("output mismatch", prompt_data["feedback"])
        self.assertEqual(prompt_data["previous_implementation"], "BAD = 1")


class EvaluationTests(unittest.TestCase):
    def test_successful_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "evaluate.py"
            solution = Path(tmp) / "solution.py"
            script.write_text(EVAL_SCRIPT)
            solution.write_text("OK = 1")
            result = run_evaluation(script, solution)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["scores"]["combined_score"], 1.0)

    def test_nonzero_exit_is_eval_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "evaluate.py"
            solution = Path(tmp) / "solution.py"
            script.write_text(EVAL_SCRIPT)
            solution.write_text("BROKEN = 1")
            result = run_evaluation(script, solution)
            self.assertEqual(result["status"], "eval_error")
            self.assertEqual(result["scores"]["combined_score"], float("-inf"))


class SelectionTests(unittest.TestCase):
    def test_best_n_last_keeps_first_iteration_samples(self):
        population = CandidatesBestNLast()
        a = SingleCandidate("a", {"combined_score": 1})
        b = SingleCandidate("b", {"combined_score": 2})
        population.update_candidates(a, current_iteration=0)
        population.update_candidates(b, current_iteration=0)
        self.assertEqual(len(population.candidates), 2)
        self.assertEqual(population.get_best()["bestN_last"].score, 2.0)

    def test_best_n_last_is_best_of_previous_generation(self):
        population = CandidatesBestNLast(max_candidates=20)
        population.update_candidates(
            SingleCandidate("a", {"combined_score": 10}), current_iteration=0
        )
        population.update_candidates(
            SingleCandidate("b", {"combined_score": 5}), current_iteration=0
        )
        self.assertEqual(population.get_best()["bestN_last"].path, "a")

        population.update_candidates(
            SingleCandidate("c", {"combined_score": 1}), current_iteration=1
        )
        self.assertEqual(population.get_best()["bestN_last"].path, "c")

    def test_best_n_all_is_best_ever(self):
        population = CandidatesBestNAll(max_candidates=20)
        population.update_candidates(
            SingleCandidate("a", {"combined_score": 10}), current_iteration=0
        )
        population.update_candidates(
            SingleCandidate("b", {"combined_score": 1}), current_iteration=1
        )
        self.assertEqual(population.get_best()["bestN_all"].path, "a")
        self.assertEqual(len(population.candidates), 2)

    def test_max_candidates_bounds_the_pool(self):
        population = CandidatesBestNAll(max_candidates=2)
        for name, score in (("a", 1), ("b", 3), ("c", 2)):
            population.update_candidates(
                SingleCandidate(name, {"combined_score": score})
            )
        self.assertEqual([item.path for item in population.candidates], ["b", "c"])

    def test_single_candidate_tolerates_missing_score(self):
        candidate = SingleCandidate("a", {"status": "eval_error"})
        self.assertEqual(candidate.score, float("-inf"))

    def test_single_candidate_custom_score_key(self):
        candidate = SingleCandidate("a", {"quality": 0.7}, score_key="quality")
        self.assertEqual(candidate.score, 0.7)

    def test_map_elites_keeps_best_per_cell(self):
        population = MapElitesCandidates(num_map_elites=2)
        population.update_candidates(
            SingleCandidate("a", {"combined_score": 1}, other="cell")
        )
        population.update_candidates(
            SingleCandidate("b", {"combined_score": 3}, other="cell")
        )
        self.assertEqual(population.get_best()["cell"].path, "b")


class FeedbackStrategyTests(unittest.TestCase):
    def test_registry_and_aliases(self):
        rules = build_feedback_strategy("rules")
        self.assertEqual(rules.name, "rules")
        self.assertFalse(rules.wants_llm)

        hybrid = build_feedback_strategy({"strategy": "rules_llm"})
        self.assertEqual(hybrid.name, "rules_llm")
        self.assertTrue(hybrid.wants_llm)
        self.assertEqual(hybrid.llm_mode, "augment")

        alias = build_feedback_strategy("classify_hint_llm")
        self.assertEqual(alias.name, "rules_llm")

        with self.assertRaises(ValueError):
            build_feedback_strategy("does-not-exist")

    def test_none_and_prompt_are_noops(self):
        for name in ("none", "prompt"):
            prompt_data = {}
            build_feedback_strategy(name).apply(
                prompt_data,
                {"status": "ok", "scores": {"combined_score": 1.0}},
                "ARTIFACT",
            )
            self.assertNotIn("feedback", prompt_data)
            self.assertEqual(prompt_data["previous_artifact"], "ARTIFACT")

    def test_evaluator_strategy_uses_evaluator_feedback(self):
        prompt_data = {}
        result = {
            "status": "ok",
            "scores": {
                "combined_score": 1.0,
                "valid": True,
                "feedback": "CUSTOM EVALUATOR HINT",
                "failure_kind": "invalid",
                "error_kind": "constraint",
            },
        }
        build_feedback_strategy("evaluator").apply(prompt_data, result, "ARTIFACT")
        self.assertEqual(prompt_data["feedback"], "CUSTOM EVALUATOR HINT")
        self.assertEqual(prompt_data["failure_stage"], INVALID)
        self.assertEqual(prompt_data["failure_kind"], "invalid")

    def test_use_feedback_upgrades_the_selected_strategy(self):
        agent = Agent(use_feedback=True, server=RecordingServer([]))
        self.assertEqual(agent.feedback_strategy.name, "rules")
        self.assertTrue(agent.feedback_strategy.wants_llm)
        self.assertEqual(agent.feedback_strategy.llm_mode, "augment")

    def test_selection_and_feedback_are_independent_config_options(self):
        config = {
            "num_iterations": 7,
            "num_samples": 2,
            "predecessor_selection_method": "bestN_last",
            "feedback_strategy": "none",
        }
        agent = Agent(prompt_config=config, server=RecordingServer([]))
        self.assertEqual(agent.num_iterations, 7)
        self.assertEqual(agent.num_samples, 2)
        self.assertEqual(agent.predecessor_selection_method, "bestN_last")
        self.assertEqual(agent.feedback_strategy.name, "none")

        # Explicit constructor arguments win over the task config.
        agent = Agent(
            prompt_config=config,
            predecessor_selection_method="map_elites",
            feedback_strategy="rules",
            num_iterations=2,
            server=RecordingServer([]),
        )
        self.assertEqual(agent.num_iterations, 2)
        self.assertEqual(agent.predecessor_selection_method, "map_elites")
        self.assertEqual(agent.feedback_strategy.name, "rules")

    def test_custom_hints_via_config(self):
        strategy = build_feedback_strategy(
            {
                "strategy": "rules",
                "profile": "generic",
                "hints": {"invalid": "CUSTOM {error_kind}"},
            }
        )
        prompt_data = {}
        strategy.apply(
            prompt_data,
            {
                "status": "ok",
                "scores": {
                    "valid": False,
                    "error_kind": "constraint",
                    "combined_score": float("-inf"),
                },
            },
            "ARTIFACT",
        )
        self.assertIn("CUSTOM constraint", prompt_data["feedback"])


class StabilityTests(unittest.TestCase):
    def test_non_finite_scores_are_rejected(self):
        self.assertEqual(score_of({"combined_score": float("nan")}), float("-inf"))
        self.assertEqual(score_of({"combined_score": float("inf")}), float("-inf"))
        candidate = SingleCandidate(
            "x", {"combined_score": float("nan")}
        )
        self.assertEqual(candidate.raw_score, float("-inf"))
        self.assertEqual(candidate.score, float("-inf"))

    def test_minimize_inverts_rank_but_not_raw_score(self):
        self.assertEqual(
            rank_score_of({"combined_score": 10}, minimize=True), -10
        )
        candidate = SingleCandidate(
            "x", {"combined_score": 10}, minimize=True
        )
        self.assertEqual(candidate.raw_score, 10)
        self.assertEqual(candidate.score, -10)

    def test_extractor_helpers(self):
        text = (
            "```python\nA=1\n```\n"
            "```python\nB=22222\n```\n"
            "```text\nC=3\n```"
        )
        self.assertEqual(match_last_code(text), "B=22222")
        self.assertEqual(match_last(text), "C=3")
        self.assertEqual(match_largest(text), "B=22222")

    def test_unified_and_json_diff_application(self):
        base = "a\nb\n"
        diff = "--- artifact\n+++ artifact\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        self.assertEqual(apply_unified_diff(base, diff), "a\nc\n")
        self.assertEqual(
            apply_json_edits(base, '{"edits":[{"old":"b","new":"c"}]}'),
            "a\nc\n",
        )
        self.assertEqual(
            apply_json_edits("x x", '{"edits":[{"old":"x","new":"y"}]}'),
            "y x",
        )
        with self.assertRaises(DiffApplyError):
            apply_json_edits(base, '{"edits":[{"old":"missing","new":"c"}]}')

    def test_map_elites_fixed_taxonomy(self):
        population = MapElitesCandidates(
            num_map_elites=2,
            method="fixed",
            taxonomy={"alpha": "first", "beta": "second"},
            classify=lambda artifact, taxonomy, method: {
                "class": "alpha",
                "new": False,
            },
        )
        key, _ = population.get_key("artifact", None)
        self.assertEqual(key, "alpha")
        population.update_candidates(
            SingleCandidate("a", {"combined_score": 1}, other=key)
        )
        self.assertEqual(population.get_best()["alpha"].path, "a")

    def test_map_elites_dynamic_taxonomy(self):
        population = MapElitesCandidates(
            num_map_elites=2,
            method="dynamic",
            taxonomy={},
            classify=lambda artifact, taxonomy, method: {
                "class": "new_class",
                "new": True,
                "description": "new thing",
            },
        )
        key, _ = population.get_key("artifact", None)
        self.assertEqual(key, "new_class")
        self.assertEqual(population.taxonomy["new_class"], "new thing")

    def test_map_elites_embedding_clustering(self):
        def embed(text):
            return [1.0, 0.0] if text.startswith("a") else [0.0, 1.0]

        population = MapElitesCandidates(
            num_map_elites=2,
            method="cluster",
            cluster_metric="embedding",
            embed=embed,
        )
        first, _ = population.get_key("aaa", None)
        same, _ = population.get_key("aaa", first)
        different, _ = population.get_key("bbb", first)
        self.assertEqual(first, same)
        self.assertNotEqual(first, different)

    @mock.patch("optimizer_agent.llm.requests.post")
    def test_llm_retries_then_raises(self, post):
        post.side_effect = requests.exceptions.ConnectionError("down")
        server = LLMServer("http://example", retries=2, retry_backoff=0)
        with self.assertRaises(LLMError):
            server.call_server([{"role": "user", "content": "hi"}], max_tokens=10)
        self.assertEqual(post.call_count, 2)

    def test_evaluator_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "evaluate.py"
            solution = Path(tmp) / "solution.py"
            script.write_text(
                "import time\ntime.sleep(5)\n"
                "print('=== Evaluation Result ===')\n"
                "print('{}')\n"
            )
            solution.write_text("X = 1")
            result = run_evaluation(script, solution, timeout=0.1)
            self.assertEqual(result["status"], "timeout")

    def test_output_is_written_to_files_and_tailed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            script = tmp / "evaluate.py"
            solution = tmp / "solution.py"
            stdout_path = tmp / "stdout.txt"
            stderr_path = tmp / "stderr.txt"
            script.write_text(
                "print('x' * 5000)\n"
                "print('=== Evaluation Result ===')\n"
                "print('{\"combined_score\": 1}')\n"
            )
            solution.write_text("X = 1")
            result = run_evaluation(
                script,
                solution,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                tail_chars=500,
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(stdout_path.exists())
            self.assertLessEqual(len(result["stdout"]), 500)


class AgentIntegrationTests(unittest.TestCase):
    def test_next_iteration_starts_from_parent_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(EVAL_SCRIPT)

            server = RecordingServer(
                ["```python\nITER_ONE = 1\n```", "```python\nITER_TWO = 2\n```"]
            )
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                predecessor_selection_method="bestN_last",
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertEqual(len(history), 2)
            first_prompt = server.calls[0][-1]["content"]
            second_prompt = server.calls[1][-1]["content"]
            self.assertIn("BASELINE", first_prompt)
            self.assertIn("ITER_ONE", second_prompt)
            self.assertNotIn("BASELINE", second_prompt)

            summary = json.loads((tmp / "experiments" / "summary.json").read_text())
            self.assertEqual(summary["best"]["score"], 1.0)

    def test_feedback_reaches_next_iteration_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(CORRECTNESS_EVAL_SCRIPT)

            server = RecordingServer(
                ["```python\nWRONG = 1\n```", "```python\nRIGHT = 1\n```"]
            )
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                predecessor_selection_method="bestN_last",
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertEqual(len(history), 2)
            second_prompt = server.calls[1][-1]["content"]
            self.assertIn("WRONG = 1", second_prompt)
            self.assertIn("incorrect", second_prompt.lower())
            self.assertIn("output mismatch", second_prompt)

    def test_generic_artifact_custom_score_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.txt"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("old artifact")
            evaluator.write_text(
                "import argparse, json\n"
                "ap = argparse.ArgumentParser()\n"
                "ap.add_argument('--solution', required=True)\n"
                "ap.add_argument('--reference', default=None)\n"
                "ap.parse_args()\n"
                "print('=== Evaluation Result ===')\n"
                "print(json.dumps({'quality': 0.75, 'valid': True}))\n"
            )

            config = {
                "system": {
                    "order": ["system"],
                    "prompts": {"system": "Improve the artifact."},
                },
                "user": {
                    "order": ["task", "artifact", "feedback"],
                    "prompts": {
                        "task": "## Task: {task_description}",
                        "artifact": "## Artifact:\n{current_artifact}",
                        "feedback": "## Feedback:\n{feedback}",
                    },
                },
                "feedback_profile": "generic",
                "feedback_hints": {"valid": "CUSTOM VALID: {metric_summary}"},
                "response_format": "raw",
                "artifact_extension": "txt",
                "score_key": "quality",
                "task_description": "Write a haiku.",
            }

            server = RecordingServer(["first artifact", "second artifact"])
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                server=server,
                prompt_config=config,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["score"], 0.75)
            self.assertTrue(history[0]["path"].endswith("core_response.txt"))
            self.assertIn("Write a haiku.", server.calls[0][-1]["content"])

            second_prompt = server.calls[1][-1]["content"]
            self.assertIn("first artifact", second_prompt)
            self.assertIn("CUSTOM VALID", second_prompt)

    def test_rules_llm_augments_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(EVAL_SCRIPT)

            server = RecordingServer(
                [
                    "```python\nFIRST = 1\n```",
                    "LLM REVIEW",
                    "```python\nSECOND = 1\n```",
                ]
            )
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                feedback_strategy="rules_llm",
                server=server,
            )
            agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            second_prompt = server.calls[2][-1]["content"]
            self.assertIn("Additional reviewer feedback", second_prompt)
            self.assertIn("LLM REVIEW", second_prompt)

    def test_evaluator_feedback_is_cleared_when_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(ONESHOT_FEEDBACK_EVAL_SCRIPT)

            server = RecordingServer(
                [
                    "```python\nFIRST = 1\n```",
                    "```python\nSECOND = 1\n```",
                    "```python\nTHIRD = 1\n```",
                ]
            )
            agent = Agent(
                num_iterations=3,
                num_samples=1,
                feedback_strategy="evaluator",
                server=server,
            )
            agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            # FIRST's feedback is correctly shown when generating SECOND, but
            # SECOND supplied no feedback, so THIRD must not inherit it.
            self.assertIn("STALE EVALUATOR HINT", server.calls[1][-1]["content"])
            self.assertNotIn("STALE EVALUATOR HINT", server.calls[-1][-1]["content"])

    def test_feedback_config_file_with_evaluator_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            feedback_config = tmp / "feedback.json"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(FEEDBACK_EVAL_SCRIPT)
            feedback_config.write_text('{"strategy": "evaluator"}')

            server = RecordingServer(["first artifact", "second artifact"])
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                feedback_config_file=str(feedback_config),
                response_format="raw",
                server=server,
            )
            agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            second_prompt = server.calls[-1][-1]["content"]
            self.assertIn("EVALUATOR HINT", second_prompt)

    def test_stage_specific_prompt_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(INVALID_STAGE_EVAL_SCRIPT)

            config = {
                "improve": {
                    "core": {
                        "system": {
                            "order": ["system"],
                            "prompts": {"system": "BASE SYSTEM"},
                        },
                        "user": {
                            "order": ["task"],
                            "prompts": {"task": "BASE TASK"},
                        },
                    }
                },
                "stage_prompts": {
                    "invalid": {
                        "system": {
                            "order": ["system"],
                            "prompts": {"system": "INVALID SYSTEM"},
                        },
                        "user": {
                            "order": ["task"],
                            "prompts": {"task": "INVALID STAGE PROMPT"},
                        },
                    }
                },
                "feedback_strategy": "evaluator",
            }

            server = RecordingServer(
                ["```python\nFIRST = 1\n```", "```python\nSECOND = 1\n```"]
            )
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                predecessor_selection_method="bestN_last",
                prompt_config=config,
                server=server,
            )
            agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertIn("BASE TASK", server.calls[0][-1]["content"])
            self.assertIn("INVALID STAGE PROMPT", server.calls[1][-1]["content"])

    def test_debug_loop_runs_for_invalid_status_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(INVALID_THEN_VALID_EVAL_SCRIPT)

            server = RecordingServer(
                ["```python\nFIRST = 1\n```", "```python\nFIXED = 1\n```"]
            )
            agent = Agent(
                num_iterations=1,
                num_samples=1,
                use_debug=True,
                num_debug_iterations=1,
                feedback_strategy="evaluator",
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertEqual(history[0]["status"], "ok")
            self.assertTrue(history[0]["path"].endswith("core_response_debug.py"))

    def test_json_diff_mode_applies_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("OLD = 1\n")
            evaluator.write_text(
                "import argparse, json, pathlib\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--solution', required=True)\n"
                "p.add_argument('--reference', default=None)\n"
                "a=p.parse_args()\n"
                "code=pathlib.Path(a.solution).read_text()\n"
                "ok='NEW' in code\n"
                "print('=== Evaluation Result ===')\n"
                "print(json.dumps({'combined_score': 1.0 if ok else -1.0, 'valid': ok, 'failure_stage': 'improve' if ok else 'invalid'}))\n"
            )
            server = RecordingServer(['{"edits":[{"old":"OLD","new":"NEW"}]}'])
            agent = Agent(
                num_iterations=1,
                num_samples=1,
                diff_based=True,
                diff_mode="json",
                feedback_strategy="evaluator",
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )
            self.assertEqual(history[0]["status"], "ok")
            self.assertIn("NEW", Path(history[0]["path"]).read_text())

    def test_minimize_selects_the_lowest_raw_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASE = 1")
            evaluator.write_text(
                "import argparse, json, pathlib\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--solution', required=True)\n"
                "p.add_argument('--reference', default=None)\n"
                "a=p.parse_args()\n"
                "code=pathlib.Path(a.solution).read_text()\n"
                "score = 10 if 'TEN' in code else 1 if 'ONE' in code else 5\n"
                "print('=== Evaluation Result ===')\n"
                "print(json.dumps({'combined_score': score, 'valid': True}))\n"
            )
            server = RecordingServer(
                ["```python\nTEN = 1\n```", "```python\nONE = 1\n```"]
            )
            agent = Agent(
                num_iterations=2,
                num_samples=1,
                minimize=True,
                predecessor_selection_method="bestN_last",
                server=server,
            )
            agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )
            summary = json.loads(
                (tmp / "experiments" / "summary.json").read_text()
            )
            self.assertEqual(summary["best"]["score"], 1)

    def test_resume_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            experiments = tmp / "experiments"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(EVAL_SCRIPT)

            server = RecordingServer(["```python\nA = 1\n```"])
            agent = Agent(num_iterations=1, num_samples=1, server=server)
            first = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(experiments),
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(len(server.calls), 1)

            for name in ("run.log", "events.jsonl", "config.json", "summary.json"):
                self.assertTrue((experiments / name).exists())

            resumed = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(experiments),
            )
            self.assertEqual(len(resumed), 1)
            self.assertEqual(len(server.calls), 1)

            overwrite_server = RecordingServer(["```python\nB = 1\n```"])
            overwrite_agent = Agent(
                num_iterations=1,
                num_samples=1,
                overwrite=True,
                server=overwrite_server,
            )
            overwrite_agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(experiments),
            )
            self.assertEqual(len(overwrite_server.calls), 1)
            config = json.loads((experiments / "config.json").read_text())
            self.assertEqual(config["search"]["predecessor_selection_method"], "bestN_last")

    def test_agent_fixed_map_elites_uses_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(EVAL_SCRIPT)
            config = {
                "improve": {
                    "core": {
                        "system": {
                            "order": ["system"],
                            "prompts": {"system": "Improve."},
                        },
                        "user": {
                            "order": ["task", "artifact"],
                            "prompts": {
                                "task": "## Task",
                                "artifact": "## Artifact:\n{current_artifact}",
                            },
                        },
                    }
                },
                "map_elites": {
                    "method": "fixed",
                    "taxonomy": {"alpha": "first kind", "beta": "second kind"},
                },
            }
            server = RecordingServer(
                [
                    "```python\nA = 1\n```",
                    '{"class": "alpha", "new": false}',
                ]
            )
            agent = Agent(
                num_iterations=1,
                num_map_elites=1,
                predecessor_selection_method="map_elites",
                prompt_config=config,
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )
            self.assertEqual(len(history), 1)
            self.assertIn("alpha", server.calls[1][-1]["content"])

    def test_debug_loop_replaces_failed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline = tmp / "baseline.py"
            evaluator = tmp / "evaluate.py"
            baseline.write_text("BASELINE = 1")
            evaluator.write_text(EVAL_SCRIPT)

            server = RecordingServer(
                ["```python\nBROKEN = 1\n```", "```python\nFIXED = 1\n```"]
            )
            agent = Agent(
                num_iterations=1,
                num_samples=1,
                use_debug=True,
                num_debug_iterations=1,
                server=server,
            )
            history = agent.run(
                baseline=str(baseline),
                evaluate=str(evaluator),
                experiments_dir=str(tmp / "experiments"),
            )

            self.assertEqual(history[0]["status"], "ok")
            self.assertTrue(history[0]["path"].endswith("core_response_debug.py"))
            self.assertEqual(history[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
