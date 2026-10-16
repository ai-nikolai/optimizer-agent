"""Turn ``prompt_data`` into new code via a single model call.

The :class:`~optimizer_agent.agent.Agent` orchestrates the search; it does not
build prompts itself. That job belongs here.

The pipeline runs one **core** stage: build chat messages from a
:class:`~optimizer_agent.prompts.PromptBuilder`, call the model, and pull the
final `` ```code ``` `` block out of the response. The same stage backs the
debug loop -- the agent simply re-calls :meth:`generate` with a bit more
context (the prior code and the error).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from optimizer_agent.utils import CORE_RESPONSE, extract_final_code_response


class GenerationPipeline:
    """Run a single model call, given a prompt builder and a model server."""

    def __init__(self, server, core_builder, *, model_params: Optional[Dict[str, Any]] = None):
        self.server = server
        self.core_builder = core_builder
        self.model_params = model_params or {}

    def _call_kwargs(self) -> Dict[str, Any]:
        allowed = {"max_tokens", "temperature", "top_p", "model"}
        return {k: v for k, v in self.model_params.items() if k in allowed}

    def _core(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.core_builder.build_messages(prompt_data)
        contents, _ = self.server.call(messages, **self._call_kwargs())
        response = contents[0] if contents else ""
        return {
            "core_prompt": messages,
            "core_response_raw": response,
            CORE_RESPONSE: extract_final_code_response(response),
        }

    def generate(self, prompt_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """One core generation. Returns ``(response_data, updated prompt_data)``."""
        out = self._core(prompt_data)
        prompt_data["new_implementation"] = out[CORE_RESPONSE]
        return out, prompt_data
