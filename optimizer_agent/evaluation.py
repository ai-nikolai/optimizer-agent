"""Run the user's evaluator as a subprocess and parse its output.

This module knows nothing about CUDA, kernels, speed-up factors, or correctness.
It only knows one thing: an evaluator is a script that prints a JSON object
after the ``=== Evaluation Result ===`` marker. Everything else -- what that
JSON means, how to turn it into a score -- is the user's business.

The :class:`EvalResult` simply carries back whatever the evaluator emitted.
The :class:`FeedbackStrategy` (see ``prompts.py``) is responsible for reading
that dict and deciding how to talk to the model.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from optimizer_agent.utils import MISSING, extract_eval_results

# Per-evaluation timeout (seconds). Kept generous: a slow evaluator should still
# get enough time, but a hung one must not strand the search.
DEFAULT_TIMEOUT = 600


@dataclass
class EvalResult:
    """The outcome of evaluating a single candidate.

    ``result`` holds the raw JSON the evaluator emitted. ``status`` is an
    internal tag (``"ok"`` when the evaluator ran and produced a parseable
    result, ``"eval_error"`` when it crashed / timed out / produced nothing
    useful). ``stdout`` / ``stderr`` carry the raw capture for logging.
    """

    status: str = "ok"
    result: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""

    @property
    def score(self) -> float:
        """The scalar the search compares. ``combined_score`` if present, else -inf."""
        from optimizer_agent.utils import score_of
        return score_of(self.result)


def evaluate_script(
    evaluate_path: str,
    solution_path: str,
    reference_path: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> EvalResult:
    """Run ``evaluate.py --solution ... [--reference ...]`` and parse its output.

    A non-zero exit code, a timeout, or any exception all collapse to a single
    failing result (``status == "eval_error"`` with ``combined_score: -inf``)
    rather than aborting the search.
    """
    command = ["python", evaluate_path, "--solution", str(solution_path)]
    if reference_path:
        command += ["--reference", str(reference_path)]
    if extra_args:
        command += list(extra_args)

    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _eval_error(status="eval_error", stderr="<timed out>")
    except Exception as exc:  # pragma: no cover - defensive
        return _eval_error(status="eval_error", stderr=f"<{exc}>")

    return parse_evaluation(proc.stdout, proc.returncode, proc.stderr)


def parse_evaluation(stdout: str, returncode: int = 0, stderr: str = "") -> EvalResult:
    """Turn raw evaluator output into an :class:`EvalResult`.

    A non-zero return code is treated as an evaluation error. A zero exit code
    is parsed for its result JSON -- and if the JSON couldn't be parsed at all
    (the ``MISSING`` sentinel) that is also treated as an evaluation error, so a
    broken evaluator can never masquerade as a working one.
    """
    result = extract_eval_results(stdout)
    if not result or result.get("__unparsed__"):
        return _eval_error(status="eval_error", stderr=stderr)
    return EvalResult(status="ok", result=result, stdout=stdout, stderr=stderr or "")


def _eval_error(*, status: str, stderr: str = "") -> EvalResult:
    """A failing result: no usable score, raw stderr for diagnostics."""
    return EvalResult(status=status, result={"combined_score": -float("inf")}, stderr=stderr)


def parse_evaluation_or_raise(stdout: str) -> Dict[str, Any]:
    """Convenience: just the parsed result dict (used by tests and simple callers)."""
    return extract_eval_results(stdout)
