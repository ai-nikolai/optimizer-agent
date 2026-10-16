"""Common prompt builder interface."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class PromptBuilder(Protocol):
    """Turn a ``prompt_data`` dict into LLM ``messages``.

    Implementations own all of the domain-specific prompt construction. The
    agent is agnostic to them, which is what lets us plug in the "simple"
    JSON-configured prompts and the KernelBench-style prompts side by side.
    """

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return the list of ``{"role": ..., "content": ...}`` messages."""
        raise NotImplementedError
