"""Prompt construction.

The agent needs to turn a ``prompt_data`` dictionary into a list of LLM
messages. Different problem domains build these messages very differently:

* The **"simple"** style is a small templating language (a JSON config that
  composes system/user sections from a handful of slots such as
  ``{current_implementation}``, ``{strategy}``, ``{feedback}``). This is the
  original optimizer-agent behaviour and stays backward compatible.

* The **KernelBench** style builds much richer prompts: it injects a reference
  architecture, backend-specific few-shot examples, GPU hardware info, and on
  every iteration after the first appends the *previous attempt* together with
  a targeted ``feedback`` message whose wording depends on *why* the candidate
  failed (compilation error, runtime error, incorrect output, or "make it
  faster") and the achieved score.

Both are exposed through the :class:`PromptBuilder` interface so the rest of the
agent never has to care which style it is using.
"""

from __future__ import annotations

from .simple import SimplePromptBuilder
from .kernel_bench import KernelBenchPromptBuilder
from .base import PromptBuilder

__all__ = [
    "PromptBuilder",
    "SimplePromptBuilder",
    "KernelBenchPromptBuilder",
]
