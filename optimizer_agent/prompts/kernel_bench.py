"""KernelBench-style prompt builder.

This is the "rich" prompt style used by robust_kernelbench. It produces prompts
of the following shape (first iteration):

    problem statement
    <few/one shot example(s): input architecture -> optimized ModelNew>
    reference architecture
    precision note
    [optional] GPU hardware guidance
    instruction: "optimize Model into ModelNew, output code only"

And on every subsequent iteration appends a **feedback** section whose wording
depends on *why* the candidate failed and how fast it got (see
``optimizer_agent.feedback.ErrorCaseFeedback``).

To stay self-contained (the agent package should not import from
``robust_kernelbench``) the TOML templates and example pairs are loaded here:

* ``kb_prompts/prompts.toml`` drives the component ordering and the backend /
  option / precision / hardware templates (mirrors the KB config exactly).
* The example source files are read by relative path from the repo root, so a
  ``kb_examples_dir`` of ``"robust_kernelbench/kb_prompts"`` (the default) works
  when the agent runs from the repo root.

This intentionally keeps the prompt *construction* here, while the heavy GPU
kernel loading / evaluation lives in ``robust_kernelbench`` (via subprocess).
"""

from __future__ import annotations

import os
import runpy
from typing import Any, Dict, List, Optional

from .base import PromptBuilder

HERE = os.path.dirname(os.path.abspath(__file__))

# Examples live in the (untracked) robust_kernelbench repo checked out next to
# optimizer_agent.
DEFAULT_KB_EXAMPLES_DIR = os.path.join("..", "robust_kernelbench", "kb_prompts")


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError:
        return ""


def _load_toml() -> Dict[str, Any]:
    """Load the KernelBench prompts.toml from the robust_kernelbench repo."""
    import tomli

    toml_path = os.path.abspath(os.path.join(HERE, "..", "robust_kernelbench", "kb_prompts", "prompts.toml"))
    with open(toml_path, "rb") as file:
        return tomli.load(file)


def _load_gpu_specs(examples_dir: str) -> Dict[str, Any]:
    """Load GPU_SPEC_INFO / GPU_DEFINITIONS / GPU_BEST_PRACTICES via runpy."""
    gpu_py = os.path.join(examples_dir, "hardware", "gpu_specs.py")
    mod = runpy.run_path(gpu_py)
    return {
        "spec_info": mod.get("GPU_SPEC_INFO", {}),
        "definitions": mod.get("GPU_DEFINITIONS", {}),
        "best_practices": mod.get("GPU_BEST_PRACTICES", []),
    }


class KernelBenchPromptBuilder(PromptBuilder):
    """Construct rich KernelBench-style prompts from a JSON config."""

    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        cfg = prompt_config or {}
        self.backend = cfg.get("backend", "cuda")
        self.option = cfg.get("option", "one_shot")  # zero_shot | one_shot | few_shot
        self.precision = cfg.get("precision", "fp32")
        self.include_hardware = cfg.get("include_hardware", True)
        self.gpu_name = cfg.get("gpu_name", "H200")
        self.custom_prompt_key = cfg.get("custom_prompt_key")
        self.examples_dir = cfg.get("kb_examples_dir", DEFAULT_KB_EXAMPLES_DIR)

        self._toml = _load_toml()
        self._gpu = _load_gpu_specs(self.examples_dir) if self.include_hardware else None

    # -- Public API -------------------------------------------------------
    def build_messages(self, prompt_data: Dict[str, Any]) -> List[Dict[str, str]]:
        system = self._system_message()
        user = self._user_message(prompt_data)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # -- Prompts ----------------------------------------------------------
    def _system_message(self) -> str:
        return (
            "You are an expert CUDA kernel engineer. You replace PyTorch "
            "operators with fast custom CUDA kernels. Output the complete "
            "optimized model named ModelNew in a single Python code block, "
            "with no prose and no testing code."
        )

    def _user_message(self, prompt_data: Dict[str, Any]) -> str:
        ref = prompt_data.get("reference") or prompt_data.get("current_implementation") or ""

        parts: List[str] = self._compose_core_blocks(ref)

        feedback = prompt_data.get("feedback")
        if feedback:
            parts.append(feedback)

        return "\n\n".join(parts).strip() + "\n"

    # -- Component composition -------------------------------------------
    def _component_order(self) -> List[str]:
        """Resolve which templates to render, and in what order.

        Honours ``custom_prompt_key`` (a ``[custom_prompts.<key>]`` block in the
        TOML) via ``components_override``, otherwise the backend/option combo.
        """
        option_data = self._toml.get("options", {}).get(self.option, {})
        components = list(option_data.get("components"))

        if self.custom_prompt_key:
            override = self._toml.get("custom_prompts", {}).get(self.custom_prompt_key, {}).get("components")
            if override:
                components = list(override)

        if self.include_hardware and "architecture" not in components:
            components.append("architecture")

        return components

    def _compose_core_blocks(self, ref: str) -> List[str]:
        backend_display = self._backend_display()
        parts: List[str] = []

        for component in self._component_order():
            rendered = self._render_component(component, ref, backend_display)
            if rendered and rendered.strip():
                parts.append(rendered)

        return parts

    def _render_component(self, component: str, ref: str, backend_display: str) -> str:
        if component == "problem_statement":
            return self._toml.get("shared", {}).get("problem_statement", "").format(backend_display=backend_display)
        if component == "architecture":
            template = self._toml.get("templates", {}).get("common", {}).get("arch_block")
            return (template or "You are given the following architecture:\n\n{ref_arch_src}").format(ref_arch_src=ref)
        if component == "precision_note":
            return f"Note: The kernels should be optimized for {self._precision_display()} precision."
        if component == "instruction":
            return self._toml.get("shared", {}).get("instruction", "").format(backend_display=backend_display)
        if component == "examples_block":
            return self._compose_examples(backend_display)
        if component.startswith("hardware_"):
            return self._render_hardware(component)
        return ""

    # -- Backend specific blocks -----------------------------------------
    def _backend_display(self) -> str:
        backend_data = self._toml.get("backends", {}).get(self.backend, {})
        return backend_data.get("backend_display", f"{self.backend.upper()} operators")

    def _precision_display(self) -> str:
        try:
            data = self._toml["precision"][self.precision]
            return data.get("precision_display", f"{self.precision.upper()} (32-bit floating point)")
        except KeyError:
            return "FP32 (32-bit floating point)"

    def _render_hardware(self, component: str) -> str:
        if not self._gpu:
            return ""
        ctx = self._gpu_context()
        if component == "hardware_header":
            return "Here is some information about the underlying hardware that you should keep in mind."
        if component == "hardware_specs":
            return (
                f"The GPU that will run the kernel is NVIDIA {ctx['gpu_name']}, "
                f"{ctx['gpu_architecture']} architecture.\n\n{ctx['gpu_specs_bullets']}"
            )
        if component == "hardware_definitions":
            return (
                "Here are some concepts about the GPU architecture that could be helpful:\n\n"
                f"{ctx['gpu_definitions_bullets']}"
            )
        if component == "hardware_best_practices":
            return (
                "Here are some best practices for writing kernels on GPU:\n\n"
                f"{ctx['gpu_best_practices_bullets']}"
            )
        return ""

    def _gpu_context(self) -> Dict[str, str]:
        spec_info = self._gpu["spec_info"]
        if self.gpu_name not in spec_info:
            raise KeyError(f"GPU name {self.gpu_name!r} not found in gpu_specs.py")

        gpu = spec_info[self.gpu_name]
        architecture = gpu.get("GPU Architecture", "Unknown")
        specs = "\n".join(f"- {v} of {k}" for k, v in gpu.items() if k != "GPU Architecture")
        defs = "\n".join(f"- {k}: {v}" for k, v in self._gpu["definitions"].items())
        best = "\n".join(f"- {x}" for x in (self._gpu["best_practices"] or []))
        return {
            "gpu_name": self.gpu_name,
            "gpu_architecture": architecture,
            "gpu_specs_bullets": specs,
            "gpu_definitions_bullets": defs,
            "gpu_best_practices_bullets": best,
        }

    def _compose_examples(self, backend_display: str) -> str:
        backend_data = self._toml.get("backends", {}).get(self.backend, {})
        option_data = self._toml.get("options", {}).get(self.option, {})
        requires = option_data.get("requires_example")
        if not requires:
            return ""

        shared = self._toml.get("shared", {})
        entry_template = (
            self._toml.get("templates", {}).get("common", {}).get("example_entry_template")
            or "{example_label}\n\nInput architecture:\n\n{input_code}\n\nOptimized with {backend_display}:\n\n{output_code}"
        )

        # Resolve example paths relative to the repo root.
        repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))

        def _resolve(rel_path: str) -> str:
            return _read_file(os.path.normpath(os.path.join(repo_root, rel_path)))

        if requires == "few_shot":
            pairs = backend_data.get("few_shot_examples") or []
            intro = "Here are examples showing how to embed custom " + backend_display + " in PyTorch:"
        else:
            # one_shot / zero_shot -> fall back to a single example.
            in_rel = backend_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
            out_rel = backend_data.get("one_shot_new_arch")
            pairs = [(in_rel, out_rel)] if (in_rel and out_rel) else []
            intro = "Here's an example to show you the syntax of inline embedding custom " + backend_display + " in PyTorch:"

        entries = []
        for i, (in_rel, out_rel) in enumerate(pairs, 1):
            entries.append(
                entry_template.format(
                    example_label=f"Example {i}:",
                    input_code=_resolve(in_rel),
                    output_code=_resolve(out_rel),
                    backend_display=backend_display,
                )
            )

        if not entries:
            return ""
        return f"{intro}\n\n" + "\n\n".join(entries)
