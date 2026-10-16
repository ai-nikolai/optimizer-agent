"""Robust evaluation.

The optimizer agent evaluates every candidate by running the user-supplied
``evaluate.py`` as a subprocess. That contract is unchanged from the original:
the evaluator takes ``--solution`` / ``--reference`` and prints a JSON object
immediately after the marker ``=== Evaluation Result ===``.

What *has* changed:

* Results are now a typed :class:`EvalResult` dataclass (``status``, ``result``,
  ``stdout`` and ``stderr``) rather than a bare tuple.
* Parsing is robust: a missing marker, malformed JSON, or a non-dict result all
  yield a *failing* result (``status == "eval_error"`` / ``{"combined_score":
  -inf}``) instead of raising and aborting the search.
* The subprocess is always run with a timeout and non-zero exits are captured as
  failures rather than propagating.
* The raw ``stdout`` / ``stderr`` of the subprocess are carried on the result so
  the agent can log them for diagnostics.
* The ``EvalResult`` accessors understand the different result shapes the
  evaluators can emit (generic ``runtime``/``runtime_original``, the robust
  KernelBench booleans, and the CUDA ``evaluate.py`` shape with
  ``runtime_custom_ms`` / ``runtime_original_ms`` / ``loading_success``).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from optimizer_agent.utils import extract_eval_results
from optimizer_agent.prompts_v2 import (
    classify_result,
    STATUS_OK,
    STATUS_EVAL_ERROR,
)

# The default timeout (seconds) for a single evaluation.
DEFAULT_EVAL_TIMEOUT = 600


@dataclass
class EvalResult:
    """A fully-typed outcome of evaluating one candidate."""

    status: str = STATUS_OK
    result: Dict[str, Any] = field(default_factory=dict)

    # Raw subprocess capture, kept for diagnostics / logging.
    stdout: str = ""
    stderr: str = ""

    # Convenient accessors ------------------------------------------------
    @property
    def combined_score(self) -> float:
        return self.result.get("combined_score", -float("inf"))

    @property
    def correctness(self) -> bool:
        return bool(self.result.get("correctness", False))

    @property
    def compiled(self) -> bool:
        return bool(self.result.get("compiled", False))

    @property
    def loading_success(self) -> bool:
        return bool(self.result.get("loading_success", True))

    @property
    def failure_class(self) -> str:
        """Score-independent class of *why* this candidate failed."""
        if self.status != STATUS_OK:
            return self.status
        return classify_result(self.result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "result": self.result,
            "combined_score": self.combined_score,
            "correctness": self.correctness,
            "compiled": self.compiled,
        }


def evaluate_script(
    evaluate_path: str,
    solution_path,
    reference_path: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    *,
    timeout: int = DEFAULT_EVAL_TIMEOUT,
) -> EvalResult:
    """Run ``evaluate.py`` on a single candidate and parse its output.

    Args:
        evaluate_path: path to the user's evaluator.
        solution_path: path (or string) to the candidate solution.
        reference_path: optional path to the reference solution.
        extra_args: any additional CLI arguments for the evaluator.
        timeout: per-evaluation timeout in seconds.

    Returns:
        A :class:`EvalResult`. Failures to run or to parse yield a failing
        result rather than an exception.
    """
    command = [
        "python",
        evaluate_path,
        "--solution", str(solution_path),
    ]
    if reference_path:
        command += ["--reference", str(reference_path)]
    if extra_args:
        command += list(extra_args)

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            status=STATUS_EVAL_ERROR,
            result={"combined_score": -float("inf"), "status": STATUS_EVAL_ERROR, "timeout": True},
        )
    except Exception as exc:
        return EvalResult(
            status=STATUS_EVAL_ERROR,
            result={"combined_score": -float("inf"), "status": STATUS_EVAL_ERROR, "error": str(exc)},
        )

    return parse_evaluation(proc.stdout, proc.returncode, proc.stderr)


def parse_evaluation(stdout: str, returncode: int = 0, stderr: str = "") -> EvalResult:
    """Parse evaluator ``stdout`` into a typed :class:`EvalResult`.

    A non-zero ``returncode`` marks the result as an evaluation error. The JSON
    result is parsed robustly (see :func:`utils.extract_eval_results`).
    """
    if returncode != 0:
        return EvalResult(
            status=STATUS_EVAL_ERROR,
            result={
                "combined_score": -float("inf"),
                "status": STATUS_EVAL_ERROR,
                "returncode": returncode,
                "stderr": (stderr or "")[:2000],
            },
            stdout="",
            stderr=stderr,
        )

    result = extract_eval_results(stdout)
    status = classify_result(result)
    return EvalResult(status=status, result=result, stdout=stdout, stderr=stderr)


def parse_evaluation_or_raise(stdout: str) -> Dict[str, Any]:
    """Convenience: just the parsed dict (used by tests and simple callers)."""
    return extract_eval_results(stdout)
