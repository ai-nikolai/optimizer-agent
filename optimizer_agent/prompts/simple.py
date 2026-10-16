"""The "simple" prompt builder.

This reproduces the original optimizer-agent prompt construction exactly: a
JSON config that, for a given ``section`` (e.g. ``improve`` or ``debug``) and
``prompt_type`` (``core`` / ``strategy`` / ``feedback``), lists a
``system`` prompt and an ordered list of ``user`` slots. Each slot is a Python
format string filled from the ``prompt_data`` dictionary, so unknown placeholders
are simply omitted.

Example (the shape used by ``examples/test_example/prompt_config.json``)::

    {
      "improve": {
        "core": {
          "system": {"order": ["system"], "prompts": {"system": "You are..."}},
          "user": {
            "order": ["task", "implementation", "strategy", "feedback"],
            "prompts": {"task": "## Task: ..."}
          }
        }
      }
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import PromptBuilder


# Default config used when no config file is provided. Kept in sync with the
# original ``utils_prompt.DEFAULT_PROMPT_CONFIG``.
DEFAULT_PROMPT_CONFIG: Dict[str, Any] = {
    "improve": {
        "core": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist."}},
            "user": {
                "order": ["task", "implementation", "strategy", "feedback"],
                "prompts": {
                    "task": "## Task: Your task is to optimize/improve the given task",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "strategy": "## Strategy: {strategy}",
                    "feedback": "## Feedback: {feedback}",
                },
            },
        },
        "strategy": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist and focus on generating the idea and plan."}},
            "user": {
                "order": ["task", "implementation", "result", "reference_result"],
                "prompts": {
                    "task": "## Task: Your task is to create an implementation plan for improving.",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "result": "## Results: The previous result was: {result}",
                    "reference_result": "## Reference result: {reference_result}",
                },
            },
        },
        "feedback": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist and focus on reviewing previous solutions and generating feedback."}},
            "user": {
                "order": ["task", "implementation", "result"],
                "prompts": {
                    "task": "## Task: Your task is to provide feedback on the implementation",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "result": "## Results: The previous result was: {result}",
                },
            },
        },
    },
    "debug": {
        "core": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist and focus on debugging."}},
            "user": {
                "order": ["task", "implementation", "strategy", "feedback"],
                "prompts": {
                    "task": "## Task: Your task is to debug the current solution",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "strategy": "## Strategy: {strategy}",
                    "feedback": "## Feedback: {feedback}",
                },
            },
        },
        "strategy": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist and focus on debugging, and specifically on generating the idea and plan."}},
            "user": {
                "order": ["task", "implementation", "error"],
                "prompts": {
                    "task": "## Task: Your task is to create an implementation plan for improving.",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "error": "## Error: {error}",
                },
            },
        },
        "feedback": {
            "system": {"order": ["system"], "prompts": {"system": "You are an expert AI scientist and focus on debugging, and specifically on reviewing previous solutions and generating feedback."}},
            "user": {
                "order": ["task", "implementation", "error"],
                "prompts": {
                    "task": "## Task: Your task is to improve the speed of the implementation",
                    "implementation": "## Current Implementation: {current_implementation}",
                    "error": "## Error: {error}",
                },
            },
        },
    },
}


class SimplePromptBuilder(PromptBuilder):
    """Config-driven prompt builder (JSON ``{"system", "order", "prompts"}``)."""

    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG

    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, str]]:
        cfg = self.prompt_config
        if not cfg:
            raise ValueError("prompt_config is empty; nothing to build from")

        messages: List[Dict[str, str]] = []

        # System prompt.
        system_cfg = cfg.get("system", {})
        system_order = system_cfg.get("order", [])
        system_prompts = system_cfg.get("prompts", {})
        system_parts = [self._fill(system_prompts.get(key), prompt_data) for key in system_order]
        system_content = " ".join(p for p in system_parts if p)
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # User prompt (composed of several ordered slots).
        user_cfg = cfg.get("user", {})
        user_order = user_cfg.get("order", [])
        user_prompts = user_cfg.get("prompts", {})
        user_parts = [self._fill(user_prompts.get(key), prompt_data) for key in user_order]
        user_parts = [p for p in user_parts if p]
        if user_parts:
            messages.append({"role": "user", "content": "\n".join(user_parts)})

        return messages

    @staticmethod
    def _fill(template: Optional[str], prompt_data: Dict[str, Any]) -> str:
        """Render a single slot, silently dropping unknown placeholders."""
        if template is None:
            return ""
        try:
            return template.format(**prompt_data)
        except KeyError:
            # Missing placeholders (e.g. no strategy yet) are simply omitted.
            result = template
            for key in list(prompt_data.keys()):
                result = result.replace("{" + key + "}", "").replace("{" + key + "}", "")
            return result
