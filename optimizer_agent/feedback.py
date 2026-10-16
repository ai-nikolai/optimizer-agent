"""Feedback strategies.

Different optimisation problems hand information back to the model very
differently. The original optimizer-agent simply re-sent a free-form ``feedback``
string. KernelBench is more structured: on every iteration after the first it
appends a message whose *wording depends on why the candidate failed* (compile
error, runtime error, incorrect output, or "make it faster") together with the
relevant error message and/or achieved score.

To support both (and more) we expose :class:`FeedbackStrategy`. Each strategy is
given the current ``prompt_data`` and the fresh ``result`` dict and returns a
dict that gets merged back into ``prompt_data``. The keys it returns are up to
the strategy, but the convention is:

* ``feedback``      - free-form feedback text (used by the simple prompt builder)
* ``error_case``    - one of the ``CASE_*`` constants (used by KB-style prompts)
* ``error_message`` - the raw error/traceback text (KB-style)
* ``previous_generation`` - the code of the candidate being iterated on (KB)
* ``score``         - the raw score (KB-style improve case)

The ``SimplePromptBuilder`` only needs ``feedback``; the
:class:`~optimizer_agent.prompts.kernel_bench.KernelBenchPromptBuilder` uses the
structured fields.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol

from .utils import classify_result


# Failure / improvement cases (KernelBench style).
CASE_EMPTY = "empty"
CASE_COMPILE_ERROR = "compile_error"
CASE_RUNTIME_ERROR = "runtime_error"
CASE_INCORRECT = "incorrect"
CASE_IMPROVE = "improve"


class FeedbackStrategy(Protocol):
    """Turn a fresh evaluation result into prompt-enriching metadata."""

    def feed_back(self, prompt_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class NoFeedback(FeedbackStrategy):
    """Produce nothing; the model is expected to keep improving blindly."""

    def feed_back(self, prompt_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class GenericFeedback(FeedbackStrategy):
    """The original simple-style feedback.

    Emits a short free-form ``feedback`` message. If the candidate produced an
    error it surfaces that; otherwise it just encourages continuation. Works with
    the simple prompt builder (``{feedback}`` slot).
    """

    def __init__(self, *, keep_previous: bool = True):
        # Whether to also forward the previous model text (helps simple prompts).
        self.keep_previous = keep_previous

    def feed_back(self, prompt_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        if not result:
            return {}

        status = classify_result(result)
        message = self._describe(status, result)
        feedback = message

        if self.keep_previous:
            previous = prompt_data.get("previous_generation")
            if previous:
                feedback = (
                    f"Here is your previous attempt:\n\n{previous}\n\n"
                    f"Please improve it. Feedback: {feedback}"
                )
        return {"feedback": feedback, "error_case": status, "error_message": result.get("main_error", "")}

    @staticmethod
    def _describe(status: str, result: Dict[str, Any]) -> str:
        return {
            CASE_COMPILE_ERROR: "Your previous answer failed to compile. Here is the error message.",
            CASE_RUNTIME_ERROR: "Your previous answer compiled but had a runtime error. Here is the error message.",
            CASE_INCORRECT: "Your previous answer ran but produced incorrect output. Here is the error message.",
        }.get(status, "Your previous answer was correct but can be improved.")


class ErrorCaseFeedback(FeedbackStrategy):
    """KernelBench structured feedback.

    The returned dict carries the coarse ``error_case`` and the raw
    ``error_message`` so the KernelBench prompt builder can render the exact
    wording that KB-style prompts expect (see
    ``get_prompt_kb_tts``/``get_prompt_kb_tts_multi_stage``). For the success
    case it computes the achieved ``score`` (speedup relative to the baseline)
    so the model knows how fast it got.
    """

    def feed_back(self, prompt_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        if not result:
            return {}

        status = classify_result(result)
        previous = prompt_data.get("previous_generation", "")
        error_message = result.get("main_error", "") or result.get("main_output", "")
        metadata = result.get("metadata", {}) or {}

        # Speedup relative to the reference baseline (used by the improve case).
        score = self._compute_score(result)

        status_msg = self._describe(status, error_message)

        feedback = "Restart your reasoning process and generate new, complete code.\n\n"
        feedback += f"{status_msg} Feedback: {prompt_data.get('previous_generation', '')}" if False else (
            f"Here is your previous attempt:\n\n{previous}\n\n"
            f"{status_msg}"
        )

        payload: Dict[str, Any] = {
            "feedback": feedback,
            "error_case": status,
            "error_message": error_message,
            "previous_generation": previous,
            "score": score,
        }
        # Surface useful correctness metadata for diagnostics.
        for key in ("max_difference", "correctness_trials"):
            if key in metadata:
                payload[key] = metadata[key]
        return payload

    @staticmethod
    def _compute_score(result: Dict[str, Any]):
        runtime_custom = result.get("runtime")
        runtime_original = result.get("runtime_original")
        if isinstance(runtime_custom, (int, float)) and isinstance(runtime_original, (int, float)):
            if runtime_custom <= 0:
                return float("inf")
            # Speedup factor: original / custom.
            return runtime_original / runtime_custom
        return result.get("combined_score")

    @staticmethod
    def _describe(status: str, error_message: str) -> str:
        return {
            CASE_COMPILE_ERROR: (
                "Your previous answer failed to compile. Here is the error message:\n\n"
                f"{error_message}"
            ),
            CASE_RUNTIME_ERROR: (
                "Your previous answer compiled successfully but had runtime errors. "
                f"Here is the error message:\n\n{error_message}"
            ),
            CASE_INCORRECT: (
                "Your previous answer was incorrect. Here is the error message:\n\n"
                f"{error_message}"
            ),
        }.get(status, "Your previous answer was correct but can be made faster.")


class ScoreFeedback(FeedbackStrategy):
    """Compare this candidate's score against a baseline/previous one.

    Useful for generic code-optimisation problems (not just kernels) where the
    reward is a scalar. It phrases the feedback relative to ``baseline_score`` or
    the previous candidate's ``score``.
    """

    def __init__(self, baseline_score: float = 0.0):
        self.baseline_score = baseline_score

    def feed_back(self, prompt_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        if not result:
            return {}

        score = result.get("combined_score")
        if not isinstance(score, (int, float)):
            score = None

        previous = prompt_data.get("previous_generation")
        prev_score = prompt_data.get("prev_score")
        if score is None or prev_score is None:
            return {}

        delta = score - prev_score
        if abs(delta) < 1e-12:
            message = "The score is unchanged from the previous attempt; try a different approach."
        elif delta > 0:
            message = f"The score improved from {prev_score:.4f} to {score:.4f} (delta +{delta:.4f})."
        else:
            message = f"The score decreased from {prev_score:.4f} to {score:.4f} (delta {delta:.4f})."

        return {
            "feedback": message,
            "error_case": CASE_IMPROVE,
            "error_message": "",
            "previous_generation": previous,
            "score": score,
        }


# Registry used by the CLI ``--feedback-strategy`` flag.
FEEDBACK_STRATEGIES = {
    "none": NoFeedback,
    "generic": GenericFeedback,
    "error_case": ErrorCaseFeedback,
    "score": lambda: ScoreFeedback(),
}


def get_feedback_strategy(name: str, **kwargs) -> FeedbackStrategy:
    if name not in FEEDBACK_STRATEGIES:
        raise ValueError(
            f"Unknown feedback strategy {name!r}. "
            f"Available: {sorted(FEEDBACK_STRATEGIES)}"
        )
    factory = FEEDBACK_STRATEGIES[name]
    if callable(factory) and not isinstance(factory, type):
        return factory(**kwargs)
    return factory()
