"""Model-call orchestration (the generation pipeline).

The :class:`optimizer_agent.agent.Agent` is agnostic to *how* prompts are built -
it just needs to turn a ``prompt_data`` dict into new candidate code via one or
more LLM calls. That job belongs here.

The pipeline runs a small, ordered set of *stages*. In the "simple" style three
stages exist (strategy -> core -> feedback), each built from a different section
(``strategy`` / ``core`` / ``feedback``) of a JSON prompt config. In the
KernelBench style only the ``core`` stage exists and it is built from a rich,
TOML-driven prompt. Because every stage is driven through the pluggable
:class:`~optimizer_agent.prompts.base.PromptBuilder` interface, the pipeline is
identical regardless of prompt style.

Two independent feedback mechanisms exist:

* a result-driven ``FeedbackStrategy`` (applied by the agent *after* evaluation,
  see :mod:`optimizer_agent.feedback`) that writes structured fields
  (``error_case``, ``score`` ...) plus a ``feedback`` string, and
* an optional LLM ``feedback`` **sub-agent** that runs here (pre-evaluation) via
  the simple-style ``feedback`` builder, mirroring the original optimizer-agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from optimizer_agent.utils import extract_final_code_response


class ResponseEnum(Enum):
    """File keys written to the experiment dir for every stage."""

    STRATEGY_PROMPT = "strategy_prompt"
    STRATEGY_RESPONSE_RAW = "strategy_response_raw"

    CORE_PROMPT = "core_prompt"
    CORE_RESPONSE_RAW = "core_response_raw"
    CORE_RESPONSE = "core_response"

    FEEDBACK_PROMPT = "feedback_prompt"
    FEEDBACK_RESPONSE_RAW = "feedback_response_raw"

    SCORES = "scores"
    EVAL_STDOUT = "eval_stdout"
    EVAL_STDERR = "eval_stderr"  # NOTE: was mislabeled "eval_stdout" in the original


class GenerationPipeline:
    """Run the ordered model calls for a single generation.

    Args:
        server: an ``LLMServer``/``FakeLLMServer`` with a ``call(messages, ...)``
            method returning ``(contents, tool_calls)``.
        core_builder: a ``PromptBuilder`` producing the *core* messages.
        strategy_builder: optional ``PromptBuilder`` for the strategy stage.
        feedback_builder: optional ``PromptBuilder`` for the feedback stage.
        use_strategy / use_feedback: whether the strategy / feedback stages run.
        model_params: extra kwargs (``temperature``, ``top_p``, ``max_tokens``,
            ``model``) forwarded to every ``server.call``.
    """

    def __init__(
        self,
        server,
        *,
        core_builder,
        strategy_builder=None,
        feedback_builder=None,
        use_strategy: bool = False,
        use_feedback: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.server = server
        self.core_builder = core_builder
        self.strategy_builder = strategy_builder
        self.feedback_builder = feedback_builder
        self.use_strategy = use_strategy and strategy_builder is not None
        self.use_feedback = use_feedback and feedback_builder is not None
        self.model_params = model_params or {}

    # -- kwargs plumbing --------------------------------------------------
    def _call_kwargs(self) -> Dict[str, Any]:
        allowed = {"max_tokens", "temperature", "top_p", "model"}
        return {k: v for k, v in self.model_params.items() if k in allowed}

    # -- individual stages ------------------------------------------------
    def _run_stage(
        self, builder, prompt_data: Dict[str, Any], key: str, sets_key: Optional[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the model for one stage and return ``(out_dict, prompt_data)``."""
        messages = builder.build_messages(prompt_data)
        contents, _ = self.server.call(messages, **self._call_kwargs())
        response = contents[0] if contents else ""
        out: Dict[str, Any] = {
            f"{key}_prompt": messages,
            f"{key}_response_raw": response,
        }
        if sets_key is not None:
            prompt_data[sets_key] = response
        return out, prompt_data

    def strategy(self, prompt_data: Dict[str, Any]):
        """Strategy stage (optional). Sets ``prompt_data["strategy"]``."""
        if not self.use_strategy:
            return {}, prompt_data
        return self._run_stage(self.strategy_builder, prompt_data, "strategy", "strategy")

    def core(self, prompt_data: Dict[str, Any]):
        """Core stage. Sets ``prompt_data["new_implementation"]`` + cleaned code."""
        out, prompt_data = self._run_stage(self.core_builder, prompt_data, "core", "new_implementation")
        out[ResponseEnum.CORE_RESPONSE.value] = extract_final_code_response(
            out[ResponseEnum.CORE_RESPONSE_RAW.value]
        )
        return out, prompt_data

    def feedback(self, prompt_data: Dict[str, Any]):
        """Feedback sub-agent stage (optional). Sets ``prompt_data["feedback"]``."""
        if not self.use_feedback:
            return {}, prompt_data
        return self._run_stage(self.feedback_builder, prompt_data, "feedback", "feedback")

    # -- composite helpers ------------------------------------------------
    def generate(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the pre-evaluation stages (strategy + core + feedback) in order."""
        out: Dict[str, Any] = {}
        strat_out, prompt_data = self.strategy(prompt_data)
        out.update(strat_out)
        core_out, prompt_data = self.core(prompt_data)
        out.update(core_out)
        feedback_out, prompt_data = self.feedback(prompt_data)
        out.update(feedback_out)
        return out, prompt_data

    def generate_debug(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a debug core generation.

        The debug workflow always plans *around the error* first, so the
        strategy stage is always enabled (it is what carries ``prompt_data['error']``
        into the core generation).
        """
        out, prompt_data = self.strategy(prompt_data)
        core_out, prompt_data = self.core(prompt_data)
        out.update(core_out)
        return out, prompt_data


if __name__ == "__main__":
    from pprint import pprint

    from optimizer_agent.llm import FakeLLMServer

    server = FakeLLMServer()
    pipeline = GenerationPipeline(server, core_builder=None, use_strategy=False)
    out, pd = pipeline.generate({})
    pprint(out)
    pprint(pd)
