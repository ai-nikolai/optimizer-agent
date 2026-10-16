"""Prompt-config loading helpers.

The agent needs a *structured* prompt config rather than the raw JSON dict, so
that the CLI can load it and the agent can build builders without repeating the
key validation. Two config shapes are supported:

* **simple** - the original ``prompt_config.json`` shape with ``improve`` /
  ``debug`` sections, each with ``core`` / ``strategy`` / ``feedback``.
* **kernel_bench** - a small dict with ``backend`` / ``option`` / ``gpu_name`` /
  ``precision`` keys.

``PromptConfig`` validates the shape, knows which path to load from (by default
``examples/test_example/prompt_config.json``), and exposes a
``to_prompt_data`` method so the CLI can hand the raw config to the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class PromptConfig:
    """A validated prompt config (simple or kernel_bench)."""

    def __init__(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        simple_config: Optional[Dict[str, Any]] = None,
        kb_config: Optional[Dict[str, Any]] = None,
        use_kb_prompt: bool = False,
    ):
        if use_kb_prompt:
            if kb_config is None:
                raise ValueError("kb_config is required when use_kb_prompt=True")
            self.path = path
            self.simple_config = {}
            self.kb_config = self._validate_kb(kb_config)
            self.use_kb_prompt = True
        else:
            self.path = path
            if simple_config is not None:
                self.simple_config = self._validate_simple(simple_config)
            else:
                self.simple_config = {}
            self.kb_config = {}
            self.use_kb_prompt = False

    @staticmethod
    def _validate_simple(config: Dict[str, Any]) -> Dict[str, Any]:
        if "improve" not in config:
            raise ValueError("simple prompt config must contain an 'improve' section")
        for section in ("improve", "debug"):
            if section not in config:
                raise ValueError(f"simple prompt config must contain a '{section}' section")
            for subsection in ("core", "strategy", "feedback"):
                if subsection not in config[section]:
                    raise ValueError(
                        f"simple prompt config [{section}] must contain a '{subsection}' section"
                    )
        return config

    @staticmethod
    def _validate_kb(config: Dict[str, Any]) -> Dict[str, Any]:
        backend = config.get("backend", "cuda")
        option = config.get("option", "one_shot")
        valid_options = ("zero_shot", "one_shot", "few_shot")
        if option not in valid_options:
            raise ValueError(
                f"KB prompt option must be one of {valid_options}, got {option!r}"
            )
        if "precision" not in config:
            config["precision"] = "fp32"
        if "gpu_name" not in config:
            config["gpu_name"] = "H200"
        return config

    def to_prompt_data(self):
        """Return the raw config dict the ``Agent`` expects."""
        if self.use_kb_prompt:
            return {
                "use_kb_prompt": True,
                "kb_prompt_config": self.kb_config,
                "improve_prompt_config": {},
                "debug_prompt_config": {},
            }
        return {
            "use_kb_prompt": False,
            "kb_prompt_config": {},
            "improve_prompt_config": self.simple_config,
            "debug_prompt_config": self.simple_config,
        }

    @classmethod
    def from_file(cls, path: Union[str, Path], use_kb_prompt: bool = False, **kb_kwargs) -> "PromptConfig":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if use_kb_prompt:
            kb_config = {**data, **kb_kwargs}
            return cls(path=path, kb_config=kb_config, use_kb_prompt=True)
        return cls(path=path, simple_config=data, use_kb_prompt=False)

    def __repr__(self) -> str:
        if self.use_kb_prompt:
            return f"PromptConfig(kernel_bench, backend={self.kb_config.get('backend', 'cuda')})"
        return f"PromptConfig(simple, path={self.path})"
