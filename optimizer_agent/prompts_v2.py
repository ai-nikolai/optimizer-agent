"""Rules + a single generic prompt builder for the optimizer agent.

The original optimizer-agent simply re-sent a free-form ``feedback`` string.
robust_kernelbench is more structured: on every iteration it appends a message
whose *wording depends on why the candidate failed* (compile error, runtime
error, incorrect output, or "make it faster") plus the achieved speed-up score.

This module brings that rule-based behaviour into the package in a single,
self-contained place. It deliberately does **not** import ``robust_kernelbench``
-- the rules live off plain evaluation result dicts, so the agent stays generic
and reusable for any optimisation problem.

The rules (mirroring robust_kernelbench's ``choose_prompt`` +
``generate_kb_tts_prompt``):

* An empty / missing result -> ``empty``.
* Not compiled -> ``compile_error``.
* Not loaded / did not run -> ``runtime_error``.
* Ran but produced the wrong answer -> ``incorrect``.
* Otherwise -> ``improve`` (it is correct; try to make it faster / better).

From the case we derive three pieces of structured feedback that get merged
into ``prompt_data``:

* ``error_case``     - the coarse classification above.
* ``error_message``  - the raw error / traceback text (if any).
* ``score``          - a speed-up factor / optimisation reward, or ``None``.

The same module also holds :class:`PromptBuilder`, a small domain-agnostic,
config-driven template renderer. It interpolates placeholders from
``prompt_data`` and silently drops slots whose placeholder is missing, so the
agent never has to know anything about the prompt content.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol, runtime_checkable


# --- evaluation result classifications --------------------------------------
STATUS_EMPTY = "empty"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_RUNTIME_ERROR = "runtime_error"
STATUS_INCORRECT = "incorrect"
STATUS_OK = "ok"
STATUS_IMPROVE = "improve"
STATUS_EVAL_ERROR = "eval_error"


def classify_result(result: Dict[str, Any]) -> str:
    """Return the coarse *case* of an evaluation result (score-independent).

    Order matters and mirrors robust_kernelbench's per-stage checks:
    compile -> runtime/load -> correctness -> otherwise improve.

    Works with either the generic result shape (``runtime``/``runtime_original``)
    or the CUDA evaluator's shape (``runtime_custom_ms``/``runtime_original_ms``).
    """
    if not result:
        return STATUS_EMPTY

    # A hard evaluation error (could not run at all) wins over everything else.
    if result.get("status") == STATUS_EVAL_ERROR or result.get("timeout"):
        return STATUS_EVAL_ERROR

    if not result.get("compiled", False):
        return STATUS_COMPILE_ERROR

    # The candidate compiled but never produced a usable model.
    if not result.get("loading_success", True):
        return STATUS_RUNTIME_ERROR

    if not result.get("correctness", False):
        return STATUS_INCORRECT

    return STATUS_IMPROVE


def extract_error_message(result: Dict[str, Any]) -> str:
    """Best-effort human-readable error text for the current result."""
    if not result:
        return ""
    for key in (
        "main_error",
        "core_error",
        "correctness_error",
        "compilation_error",
        "runtime_error",
        "loading_error",
        "error",
    ):
        msg = result.get(key)
        if msg:
            return str(msg)
    out = result.get("main_output")
    if out:
        return str(out)
    return ""


def _runtimes(result: Dict[str, Any]):
    """Return ``(runtime, runtime_original)`` from any supported key shape.

    Recognised shapes (first match wins):

    * ``runtime_custom_ms`` / ``runtime_original_ms`` -- the CUDA
      ``evaluate.py`` scorer.
    * ``runtime`` / ``runtime_original`` -- robust_kernelbench / generic.
    * ``runtime_ms`` / ``runtime_original_ms`` -- an alternate ms shape.
    """
    def num(key):
        val = result.get(key)
        return val if isinstance(val, (int, float)) else None

    runtime = num("runtime_custom_ms")
    runtime_original = num("runtime_original_ms")
    if runtime is None and runtime_original is None:
        runtime = num("runtime")
        runtime_original = num("runtime_original")
    if runtime is None and runtime_original is None:
        runtime = num("runtime_ms")
        runtime_original = num("runtime_original_ms")
    return runtime, runtime_original


def compute_score(result: Dict[str, Any]) -> Any:
    """Return a scalar "score" for the result, or ``None`` if unknown.

    Preference order (covers the different evaluators):

    1. an explicit ``combined_score`` field (the CUDA evaluate.py already
       stores ``runtime_original_ms - runtime_custom_ms``),
    2. speed-up factor ``runtime_original / runtime`` (both positive).
    """
    if not result:
        return None

    # If the evaluator already produced a combined score, use it directly
    # rather than recomputing a ratio.
    combined = result.get("combined_score")
    if isinstance(combined, (int, float)):
        return combined

    runtime, runtime_original = _runtimes(result)
    if runtime is not None and runtime_original is not None:
        if runtime > 0:
            # Speed-up factor ("bigger is faster").
            return runtime_original / runtime
        if runtime_original > 0:
            return float("inf")
        # A fast idle kernel beats a slow baseline by a fixed amount.
        return 0.0

    return None


# --- feedback strategies -----------------------------------------------------
@runtime_checkable
class FeedbackStrategy(Protocol):
    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class NoFeedback(FeedbackStrategy):
    """Produce nothing; the model is expected to keep improving blindly."""

    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {}


#: The case -> wording hints. Kept short and neutral so the builder stays
#: domain-agnostic while still carrying the structured case/error/score fields
#: that make the feedback meaningful.
_CASE_HINTS = {
    STATUS_COMPILE_ERROR: "Your previous answer failed to compile. See the error below.",
    STATUS_RUNTIME_ERROR: "Your previous answer failed to run. See the error below.",
    STATUS_INCORRECT: "Your previous answer was incorrect. See the details below.",
    STATUS_EMPTY: "Your previous answer was empty or could not be evaluated.",
    STATUS_EVAL_ERROR: "The candidate could not be evaluated. See the error below.",
    STATUS_IMPROVE: "Your previous answer was correct. Try to make it faster or better.",
}


class GenericFeedbackStrategy(FeedbackStrategy):
    """The single, rule-based feedback path for the agent.

    Given a fresh evaluation ``result`` it writes the structured feedback
    fields (``error_case``, ``error_message``, ``score``, ``previous_generation``,
    ``prev_score``) plus a short ``feedback`` string. The agent merges the
    returned dict back into ``prompt_data`` before the next iteration.
    """

    def __init__(self, *, keep_previous: bool = True):
        self.keep_previous = keep_previous

    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not result:
            return {}

        case = classify_result(result)
        error_message = extract_error_message(result)
        score = compute_score(result)
        previous = prompt_data.get("previous_generation", "")

        if self.keep_previous and previous:
            feedback = (
                f"Here is your previous attempt:\n\n{previous}\n\n"
                f"{_CASE_HINTS.get(case, '')}\n\n"
                f"Feedback: {error_message}".rstrip()
            )
        else:
            feedback = _CASE_HINTS.get(case, "")
            if error_message:
                feedback = f"{feedback}\n\nError: {error_message}".strip()

        payload: Dict[str, Any] = {
            "error_case": case,
            "error_message": error_message,
            "score": score,
            "feedback": feedback,
            "previous_generation": previous,
        }
        if score is not None:
            payload["prev_score"] = prompt_data.get("prev_score", score)
        metadata = result.get("metadata", {}) or {}
        for key in ("max_difference", "correctness_trials"):
            if key in metadata:
                payload[key] = metadata[key]
        return payload


# --- registry ----------------------------------------------------------------
FEEDBACK_STRATEGIES = {
    "none": NoFeedback,
    "v2": GenericFeedbackStrategy,
}


def get_feedback_strategy(name: str = "v2", **kwargs) -> FeedbackStrategy:
    """Return a feedback-strategy instance for ``name`` (default ``"v2"``)."""
    if name not in FEEDBACK_STRATEGIES:
        raise ValueError(
            f"Unknown feedback strategy {name!r}. Available: {sorted(FEEDBACK_STRATEGIES)}"
        )
    factory = FEEDBACK_STRATEGIES[name]
    if callable(factory) and not isinstance(factory, type):
        return factory(**kwargs)
    return factory()


# --- generic prompt builder --------------------------------------------------
class PromptBuilder:
    """A small, domain-agnostic, config-driven template renderer.

    A config is ``{"system": {order, prompts}, "user": {order, prompts}}``. Each
    slot value is a format string interpolated from ``prompt_data``; placeholders
    that are absent are simply omitted (so a slot that needs e.g. ``{feedback}``
    disappears on the first iteration, before feedback exists). This keeps the
    agent completely agnostic to the *content* of the prompt it emits.

    The bundled :data:`DEFAULT_PROMPT_CONFIG` is deliberately neutral (it works
    for any problem): the model is told to replace a reference implementation
    with an optimized ``ModelNew``. Swap in another config to change the wording
    without touching the agent.
    """

    def __init__(self, prompt_config: Dict[str, Any] | None = None):
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, str]]:
        cfg = self.prompt_config or {}
        messages = []
        for role in ("system", "user"):
            cfg_section = cfg.get(role, {})
            order = cfg_section.get("order", [])
            prompts = cfg_section.get("prompts", {})
            parts = [self._fill(prompts.get(key), prompt_data) for key in order]
            parts = [p for p in parts if p]
            if parts:
                joined = "\n".join(parts) if role == "user" else " ".join(parts)
                messages.append({"role": role, "content": joined})
        return messages

    @staticmethod
    def _fill(template: str | None, prompt_data: Dict[str, Any]) -> str:
        """Render a slot, dropping the whole slot if any placeholder is missing.

        This keeps the agent generic: on the first iteration ``feedback`` does
        not exist, so the ``## Feedback:`` slot simply disappears rather than
        emitting an empty ``## Feedback: `` heading.
        """
        if template is None:
            return ""
        try:
            return template.format(**prompt_data)
        except KeyError:
            return ""


#: The single, neutral prompt used by the agent. Domain-agnostic: swap the
#: strings here to adapt to a different optimisation problem without touching
#: the agent core.
DEFAULT_PROMPT_CONFIG: Dict[str, Any] = {
    "system": {
        "order": ["system"],
        "prompts": {
            "system": "You are an expert optimisation engineer. Replace the reference implementation with a faster 'ModelNew' implementation that preserves its behaviour, and output the complete code only."
        },
    },
    "user": {
        "order": ["task", "implementation", "reference", "feedback"],
        "prompts": {
            "task": "## Task: Optimise the reference implementation below.",
            "implementation": "## Current Implementation:\n\n{current_implementation}",
            "reference": "## Reference Implementation:\n\n{reference}",
            "feedback": "## Feedback:\n\n{feedback}",
        },
    },
}
