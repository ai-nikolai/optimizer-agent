"""Prompt configuration and rendering.

The agent is agnostic to prompt *content*: a config is a set of named stages
(``core``, ``strategy``, ``feedback``), each with ``system`` and ``user``
sections. A slot whose ``{placeholder}`` is missing from ``prompt_data`` is
dropped, so the same config works before and after a result exists.

Two config shapes are accepted:

- nested (the historical shape): ``{"improve": {...}, "debug": {...}}``
- flat: ``{"system": ..., "user": ...}`` -- used for both improve and debug

The built-in config is deliberately **domain-agnostic**: it talks about an
``artifact``, not a Python implementation. Code/kernel examples override it
with their own JSON configs. Optional top-level keys on a config also control
the generic behaviour:

    {
      "feedback_profile": "generic" | "kernel" | {...},
      "feedback_hints": {"invalid": "...", ...},
      "response_format": "code" | "raw" | "json",
      "artifact_extension": "py",
      "score_key": "combined_score",
      "task_description": "..."
    }
"""

from optimizer_agent.feedback import normalise_stage

DEFAULT_PROMPT_CONFIG = {
    "improve": {
        "core": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert optimizer. Improve the given artifact. "
                        "Return the complete artifact in the requested format."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": (
                        "## Task: Optimize/improve the artifact while preserving "
                        "its required behaviour."
                    ),
                    "task_description": "## Task details:\n{task_description}",
                    "reference": "## Reference Artifact:\n{reference}",
                    "artifact": "## Current Artifact:\n{current_artifact}",
                    "result": "## Result of the current artifact:\n{result}",
                    "failure_kind": "## Previous outcome: {failure_kind}",
                    "failure_stage": "## Previous failure stage: {failure_stage}",
                    "strategy": "## Strategy:\n{strategy}",
                    "feedback": "## Feedback:\n{feedback}",
                },
                "order": [
                    "task",
                    "task_description",
                    "reference",
                    "artifact",
                    "result",
                    "failure_kind",
                    "failure_stage",
                    "strategy",
                    "feedback",
                ],
            },
        },
        "strategy": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert planner. Produce a short, concrete plan "
                        "for improving the artifact."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task: Propose an improvement plan.",
                    "task_description": "## Task details:\n{task_description}",
                    "artifact": "## Current Artifact:\n{current_artifact}",
                    "result": "## Current result:\n{result}",
                    "reference": "## Reference Artifact:\n{reference}",
                },
                "order": ["task", "task_description", "artifact", "result", "reference"],
            },
        },
        "feedback": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert reviewer. Explain what to change next, "
                        "using the result and the candidate artifact."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task: Give concise feedback for the next attempt.",
                    "artifact": "## Candidate Artifact:\n{new_artifact}",
                    "result": "## Result:\n{result}",
                    "failure_kind": "## Outcome:\n{failure_kind}",
                    "failure_stage": "## Failure stage:\n{failure_stage}",
                    "hint": "## Automatic diagnosis:\n{feedback_hint}",
                    "error": "## Error:\n{error}",
                    "reference": "## Reference Artifact:\n{reference}",
                },
                "order": [
                    "task",
                    "artifact",
                    "result",
                    "failure_kind",
                    "failure_stage",
                    "hint",
                    "error",
                    "reference",
                ],
            },
        },
    },
    "debug": {
        "core": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert debugger. Fix the artifact so it is "
                        "accepted by the evaluator. Return the complete artifact "
                        "in the requested format."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task: Debug the current artifact.",
                    "task_description": "## Task details:\n{task_description}",
                    "artifact": "## Current Artifact:\n{current_artifact}",
                    "error": "## Error:\n{error}",
                    "failure_stage": "## Failure stage:\n{failure_stage}",
                    "result": "## Result:\n{result}",
                    "strategy": "## Strategy:\n{strategy}",
                    "feedback": "## Feedback:\n{feedback}",
                    "reference": "## Reference Artifact:\n{reference}",
                },
                "order": [
                    "task",
                    "task_description",
                    "artifact",
                    "error",
                    "failure_stage",
                    "result",
                    "strategy",
                    "feedback",
                    "reference",
                ],
            },
        },
        "strategy": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert debugger. Propose a short plan for "
                        "fixing the failure."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task: Propose a debugging plan.",
                    "artifact": "## Current Artifact:\n{current_artifact}",
                    "error": "## Error:\n{error}",
                    "reference": "## Reference Artifact:\n{reference}",
                },
                "order": ["task", "artifact", "error", "reference"],
            },
        },
        "feedback": {
            "system": {
                "prompts": {
                    "system": (
                        "You are an expert debugger. Explain the likely cause and "
                        "what to fix next."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task: Give concise debugging feedback.",
                    "artifact": "## Candidate Artifact:\n{new_artifact}",
                    "error": "## Error:\n{error}",
                    "failure_kind": "## Outcome:\n{failure_kind}",
                    "failure_stage": "## Failure stage:\n{failure_stage}",
                    "hint": "## Automatic diagnosis:\n{feedback_hint}",
                    "result": "## Result:\n{result}",
                    "reference": "## Reference Artifact:\n{reference}",
                },
                "order": [
                    "task",
                    "artifact",
                    "error",
                    "failure_kind",
                    "failure_stage",
                    "hint",
                    "result",
                    "reference",
                ],
            },
        },
    },
    "classify": {
        "core": {
            "system": {
                "prompts": {
                    "system": (
                        "You are a taxonomy classifier. Assign the artifact to a "
                        "class and return only the requested JSON."
                    )
                },
                "order": ["system"],
            },
            "user": {
                "prompts": {
                    "task": "## Task\nAssign the artifact to a class.",
                    "task_description": "## Task description:\n{task_description}",
                    "mode": "## Mode:\n{mode_instruction}",
                    "taxonomy": "## Existing classes:\n{taxonomy_text}",
                    "artifact": "## Artifact:\n{artifact}",
                    "output": (
                        "## Output\nReturn JSON only: "
                        '{"class": "class_name", "new": false, '
                        '"description": "short description"}'
                    ),
                },
                "order": [
                    "task",
                    "task_description",
                    "mode",
                    "taxonomy",
                    "artifact",
                    "output",
                ],
            },
        }
    },
}


def resolve_prompt_config(config, section="improve"):
    """Return the stage bundle (``core``/``strategy``/``feedback``) for a run.

    Supports the nested historical shape, the flat shape, and ``None``.
    """
    if config is None:
        return DEFAULT_PROMPT_CONFIG[section]
    if section in config:
        return config[section]
    if section == "classify":
        # A flat improve/user config is not a classifier prompt.
        return DEFAULT_PROMPT_CONFIG["classify"]
    if "system" in config or "user" in config:
        return config  # flat config: use for improve and debug alike
    return DEFAULT_PROMPT_CONFIG[section]


def resolve_stage_configs(config, section="improve"):
    """Return ``{failure_stage: prompt_bundle}`` overrides from a config.

    A config may contain ``stage_prompts`` (for ``improve``) or
    ``debug_stage_prompts`` (for ``debug``). Each value is either a full stage
    bundle with a ``core`` key or a single core block; missing strategy/feedback
    sections are inherited from the base bundle.
    """
    base = resolve_prompt_config(config, section)
    if not isinstance(config, dict):
        return {}
    key = "stage_prompts" if section == "improve" else "debug_stage_prompts"
    overrides = config.get(key) or {}
    resolved = {}
    for stage, block in overrides.items():
        if not isinstance(block, dict):
            continue
        bundle = dict(base)
        if "core" in block:
            bundle.update(block)
        else:
            bundle["core"] = block
        resolved[normalise_stage(stage) or str(stage)] = bundle
    return resolved


def create_prompt(prompt_type="core", prompt_data=None, prompt_config=None):
    """Build the chat ``messages`` list for one stage of one config bundle."""
    prompt_data = prompt_data or {}
    config = prompt_config or DEFAULT_PROMPT_CONFIG["improve"]
    cfg = config.get(prompt_type)
    if cfg is None and ("system" in config or "user" in config):
        # Flat config: the same system/user block backs every stage.
        cfg = config
    if cfg is None:
        raise ValueError(f"Prompt stage '{prompt_type}' not found in config.")

    messages = []
    for role in ("system", "user"):
        section = cfg.get(role, {})
        prompts = section.get("prompts", {})
        parts = []
        for key in section.get("order", []):
            template = prompts.get(key)
            if not template:
                continue
            try:
                rendered = template.format(**prompt_data)
            except (KeyError, IndexError, ValueError):
                # A slot whose data is unavailable is simply omitted.
                continue
            parts.append(rendered)
        if parts:
            separator = " " if role == "system" else "\n"
            messages.append({"role": role, "content": separator.join(parts)})

    return messages
