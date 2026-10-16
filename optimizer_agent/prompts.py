"""Generic, rule-based feedback + a config-driven prompt builder.

The original optimizer-agent just re-sent a free-form ``feedback`` string. That
works, but a more useful agent tells the *model why* its last attempt failed,
then feeds back whatever numeric signal the evaluator produced.

This module keeps that behaviour but makes it **domain-agnostic**: it operates
on a plain result dict (whatever ``evaluate.py`` emitted) and never mentions
CUDA, kernels, speed-up factors, "compiled", "correctness", or a concrete
problem. It does the minimum that is true of *any* optimisation problem:

1. **classify** the result -- did it fail to run? is it wrong? or is it fine
   and we just want a bigger number? Expressed only through generic signals a
   result dict may carry (``status``, ``timeout``, ``correctness``,
   ``combined_score``).
2. **score** the result -- the number the search compares (``combined_score``).
3. **summarise** -- a short line the agent merges back into ``prompt_data``.

Two pieces ship here:

* the :class:`FeedbackStrategy` family, and
* :class:`PromptBuilder`, a tiny, config-driven renderer that turns
  ``prompt_data`` into chat messages without the agent caring about content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

# --- result classifications (score-independent) ----------------------------
STATUS_EMPTY = "empty"
STATUS_EVAL_ERROR = "eval_error"
STATUS_INCORRECT = "incorrect"
STATUS_IMPROVE = "improve"

#: Coarse case -> short, neutral, domain-agnostic wording.
_CASE_HINTS = {
    STATUS_EVAL_ERROR: "Your previous attempt could not be evaluated. See the details below.",
    STATUS_INCORRECT: "Your previous attempt was incorrect. See the details below.",
    STATUS_IMPROVE: "Your previous attempt was accepted. Try to improve the score.",
}


def classify_result(result: Dict[str, Any]) -> str:
    """Return the coarse *why* of a result (score-independent).

    Order matters: an evaluation error beats a wrong answer, which beats an
    accepted-but-improvable answer. Anything with no recognisable failure
    signal falls through to ``improve`` so the loop keeps trying.
    """
    if not result:
        return STATUS_EMPTY
    if result.get("status") == STATUS_EVAL_ERROR or result.get("timeout"):
        return STATUS_EVAL_ERROR
    # A generic correctness flag (boolean) that is false -> the answer is wrong.
    if not result.get("correctness", True):
        return STATUS_INCORRECT
    return STATUS_IMPROVE


def _summarise(result: Dict[str, Any]) -> str:
    """A short, readable summary of what the evaluator reported."""
    for key in ("error", "error_message", "main_error"):
        msg = result.get(key)
        if msg:
            return str(msg)
    return str(result.get("main_output") or "")


def compute_score(result: Dict[str, Any]) -> Any:
    """The reward signal, or ``None`` if the evaluator did not provide one."""
    if not result:
        return None
    combined = result.get("combined_score")
    if isinstance(combined, (int, float)):
        return float(combined)
    return None


# --- feedback strategies ---------------------------------------------------
@runtime_checkable
class FeedbackStrategy(Protocol):
    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class NoFeedback(FeedbackStrategy):
    """Produce nothing -- the model keeps improving blindly."""

    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {}


class GenericFeedbackStrategy(FeedbackStrategy):
    """The single, rule-based feedback path.

    Given the latest result it writes:

    * ``error_case``     -- the coarse classification above,
    * ``feedback``       -- a short, readable line (hint + summary/score),
    * ``score``          -- the reward the evaluator reported, or ``None``,
    * ``previous_generation`` -- the model's own prior attempt, for context.

    The strategy owns ``previous_generation``/``prev_score`` (read directly from
    ``prompt_data``, ``new_implementation``) and the agent simply merges the
    returned dict back in.
    """

    def feed_back(
        self, prompt_data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not result:
            return {}

        case = classify_result(result)
        summary = _summarise(result)
        score = compute_score(result)
        previous = prompt_data.get("new_implementation", "")

        if previous:
            feedback = (
                f"Here is your previous attempt:\n\n{previous}\n\n"
                f"{_CASE_HINTS.get(case, '')}\n\nSummary: {summary}".rstrip()
            )
        else:
            feedback = _CASE_HINTS.get(case, "")
            if summary:
                feedback = f"{feedback}\n\nSummary: {summary}".strip()

        payload: Dict[str, Any] = {
            "error_case": case,
            "feedback": feedback,
            "score": score,
            "previous_generation": previous,
        }
        if score is not None:
            payload["prev_score"] = prompt_data.get("prev_score", score)
        return payload


#: Strategy registry, selected via the ``--feedback-strategy`` CLI flag.
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


# --- generic prompt builder ------------------------------------------------
#: The one, neutral prompt by default. Swap the strings here (or pass a config
#: file via ``--prompt-config-file``) to change the wording without touching
#: the agent. It works for *any* optimisation problem: replace the reference
#: implementation with a faster ``ModelNew`` that preserves its behaviour.
DEFAULT_PROMPT_CONFIG: Dict[str, Any] = {
    "system": {
        "order": ["system"],
        "prompts": {
            "system": (
                "You are an expert optimisation engineer. Replace the reference "
                "implementation with a faster 'ModelNew' implementation that "
                "preserves its behaviour, and output the complete code only."
            )
        },
    },
    "user": {
        "order": ["task", "implementation", "feedback"],
        "prompts": {
            "task": "## Task: Optimise the reference implementation below.",
            "implementation": "## Current Implementation:\n\n{current_implementation}",
            "feedback": "## Feedback:\n\n{feedback}",
        },
    },
}


class PromptBuilder:
    """A tiny, config-driven chat-message renderer.

    A config is ``{"system": {order, prompts}, "user": {order, prompts}}``. Each
    slot is a format string interpolated from ``prompt_data``; a slot whose
    placeholder is missing is **silently dropped** (so ``## Feedback:``
    disappears on the first iteration, before feedback exists). The agent is
    therefore completely agnostic to the prompt content.
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
        """Render a slot; drop it entirely if any placeholder is missing."""
        if template is None:
            return ""
        try:
            return template.format(**prompt_data)
        except KeyError:
            return ""
