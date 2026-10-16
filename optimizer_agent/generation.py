"""Model-call orchestration (the generation pipeline).

The :class:`optimizer_agent.agent.Agent` is agnostic to *how* prompts are built -
it just needs to turn a ``prompt_data`` dict into new candidate code via the
LLM. That job belongs here.

The pipeline runs a single **core** stage built from a generic
:class:`~optimizer_agent.prompts_v2.PromptBuilder`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from optimizer_agent.utils import extract_final_code_response


class GenerationPipeline:
    """Run the model call for a single generation.

    Args:
        server: an ``LLMServer``/``FakeLLMServer`` with a ``call(messages, ...)``
            method returning ``(contents, tool_calls)``.
        core_builder: a ``PromptBuilder`` producing the *core* messages.
        model_params: extra kwargs (``temperature``, ``top_p``, ``max_tokens``,
            ``model``) forwarded to ``server.call``.
    """

    def __init__(
        self,
        server,
        *,
        core_builder,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.server = server
        self.core_builder = core_builder
        self.model_params = model_params or {}

    # -- kwargs plumbing --------------------------------------------------
    def _call_kwargs(self) -> Dict[str, Any]:
        allowed = {"max_tokens", "temperature", "top_p", "model"}
        return {k: v for k, v in self.model_params.items() if k in allowed}

    # -- core stage ------------------------------------------------------
    def core(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Core stage. Sets ``prompt_data["new_implementation"]`` + cleaned code."""
        messages = self.core_builder.build_messages(prompt_data)
        contents, _ = self.server.call(messages, **self._call_kwargs())
        response = contents[0] if contents else ""
        out: Dict[str, Any] = {
            "core_prompt": messages,
            "core_response_raw": response,
        }
        prompt_data["new_implementation"] = response
        out["core_response"] = extract_final_code_response(response)
        return out, prompt_data

    def generate(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the pre-evaluation stages in order (core, for now)."""
        out, prompt_data = self.core(prompt_data)
        return out, prompt_data

    def generate_debug(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a debug core generation (planning around the captured error)."""
        out, prompt_data = self.core(prompt_data)
        return out, prompt_data


if __name__ == "__main__":
    from pprint import pprint

    from optimizer_agent.llm import FakeLLMServer
    from optimizer_agent.prompts_v2 import PromptBuilder

    server = FakeLLMServer()
    pipeline = GenerationPipeline(server, core_builder=PromptBuilder(), model_params={})
    out, pd = pipeline.generate({"current_implementation": "def f(): return 1"})
    pprint(out)
    pprint(pd)
