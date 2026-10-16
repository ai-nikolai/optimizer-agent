"""Lightweight, GPU-free unit checks for the optimizer agent.

These exercise prompt construction (both the "simple" JSON style and the
KernelBench / TOML style) and the pluggable feedback strategies. They never
touch torch / CUDA, so they run anywhere Python 3.9+ is installed.

Run with either ``pytest`` or ``python -m optimizer_agent.tests.test_units``.

The KernelBench tests require the KB prompt assets (``kb_prompts/prompts.toml``,
``kb_prompts/hardware/gpu_specs.py`` and the example ``*.py`` files). Those live
in the companion ``robust_kernelbench`` checkout. When they are not present the
KB tests are skipped rather than failing, so this suite still runs in a
minimal install.
"""

from __future__ import annotations

from optimizer_agent.prompts.simple import SimplePromptBuilder
from optimizer_agent.feedback import (
    NoFeedback,
    GenericFeedback,
    ErrorCaseFeedback,
    ScoreFeedback,
    get_feedback_strategy,
)


def _kb_assets_available() -> bool:
    """True if the KernelBench prompt assets can be located."""
    from optimizer_agent.prompts.kernel_bench import (
        _find_kb_prompts_dir,
        _load_gpu_specs,
    )

    kb_dir = _find_kb_prompts_dir()
    if kb_dir is None:
        return False
    try:
        _load_gpu_specs(kb_dir)
        return True
    except Exception:
        return False


def _kb_config(option="one_shot"):
    return {
        "backend": "cuda",
        "option": option,
        "gpu_name": "L40S",  # a GPU that exists in robust_kernelbench/kb_prompts/hardware/gpu_specs.py
        "precision": "fp32",
        "include_hardware": True,
    }


def _make_kb(option="one_shot"):
    from optimizer_agent.prompts.kernel_bench import KernelBenchPromptBuilder

    return KernelBenchPromptBuilder(_kb_config(option))


# ----------------------------------------------------------------------
# Prompt building: simple style
# ----------------------------------------------------------------------
def test_simple_builder_core():
    cfg = {
        "system": {"order": ["system"], "prompts": {"system": "You are {who}."}},
        "user": {"order": ["task", "impl"], "prompts": {
            "task": "## Task",
            "impl": "## Implementation: {current_implementation}",
        }},
    }
    b = SimplePromptBuilder(cfg)
    messages = b.build_messages({"current_implementation": "def f(): pass", "who": "an expert"})
    roles = {m["role"]: m["content"] for m in messages}
    assert roles["system"] == "You are an expert."
    assert "def f(): pass" in roles["user"]


def test_simple_builder_fills_present_placeholders():
    """Present placeholders are substituted; a present system renders."""
    cfg = {
        "system": {"order": ["system"], "prompts": {"system": "SYS"}},
        "user": {"order": ["impl"], "prompts": {"impl": "{current_implementation}"}},
    }
    b = SimplePromptBuilder(cfg)
    messages = b.build_messages({"current_implementation": "def f(): return 1"})
    roles = {m["role"]: m["content"] for m in messages}
    assert roles["user"] == "def f(): return 1"


def test_default_core_section_builds():
    """The bundled DEFAULT_PROMPT_CONFIG 'improve.core' section renders."""
    from optimizer_agent.prompts.simple import DEFAULT_PROMPT_CONFIG

    b = SimplePromptBuilder(DEFAULT_PROMPT_CONFIG["improve"]["core"])
    messages = b.build_messages({"current_implementation": "x"})
    assert any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" for m in messages)
    assert "x" in messages[-1]["content"]


# ----------------------------------------------------------------------
# Prompt building: KernelBench style (TOML-driven)
# ----------------------------------------------------------------------
def test_kb_one_shot_includes_example_and_arch():
    b = _make_kb("one_shot")
    user = b.build_messages({"reference": "class Model:\n    def get_inputs(self):\n        return []\n"})[1]["content"]
    assert "Here" in user  # intro line
    assert "def get_inputs" in user  # reference architecture block


def test_kb_few_shot_includes_multiple_examples():
    b = _make_kb("few_shot")
    user = b.build_messages({"reference": "class Model:\n    pass\n"})[1]["content"]
    assert "Example 1:" in user
    assert "Example 2:" in user


def test_kb_zero_shot_has_no_examples():
    b = _make_kb("zero_shot")
    user = b.build_messages({"reference": "class Model:\n    pass\n"})[1]["content"]
    # zero-shot option has no example components.
    assert "Example 1:" not in user


def test_kb_backend_display_reflected():
    b = _make_kb("one_shot")
    user = b.build_messages({"reference": "class Model:\n    pass\n"})[1]["content"]
    assert "CUDA operators" in user


def test_kb_rejects_unknown_gpu():
    from optimizer_agent.prompts.kernel_bench import KernelBenchPromptBuilder

    b = _make_kb("one_shot")
    b._gpu_context()  # L40S should exist in the default gpu specs
    bad = KernelBenchPromptBuilder({**_kb_config(), "gpu_name": "NOT_A_GPU"})
    try:
        bad._gpu_context()
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown GPU")


# ----------------------------------------------------------------------
# Feedback strategies
# ----------------------------------------------------------------------
def test_no_feedback_returns_empty():
    out = NoFeedback().feed_back({"previous_generation": "x"}, {"combined_score": 0.5})
    assert out == {}


def test_generic_feedback_describes_errors():
    result = {"compiled": True, "correctness": True, "runtime_success": False, "main_error": "boom"}
    out = GenericFeedback().feed_back({"previous_generation": "prev"}, result)
    assert out["error_case"] == "runtime_error"
    # The actual error text is surfaced via error_message, not embedded in feedback.
    assert out["error_message"] == "boom"
    assert "previous attempt" in out["feedback"].lower()


def test_error_case_feedback_includes_previous_and_score():
    result = {
        "runtime": 1.0,
        "runtime_original": 4.0,
        "compiled": True,
        "correctness": True,
        "main_error": "",
        "metadata": {"max_difference": 0.1},
    }
    out = ErrorCaseFeedback().feed_back({"previous_generation": "my code"}, result)
    assert out["error_case"] == "ok"
    assert out["previous_generation"] == "my code"
    assert out["score"] == 4.0  # runtime_original / runtime
    assert "max_difference" in out


def test_score_feedback_compares_prev_score():
    prev = {"prev_score": 10.0, "previous_generation": "a"}
    out = ScoreFeedback().feed_back(prev, {"combined_score": 12.0})
    assert "improved" in out["feedback"]
    assert out["score"] == 12.0


def test_score_feedback_returns_empty_without_prev_score():
    out = ScoreFeedback().feed_back({}, {"combined_score": 12.0})
    assert out == {}


def test_registry_produces_expected_types():
    assert isinstance(get_feedback_strategy("none"), NoFeedback)
    assert isinstance(get_feedback_strategy("generic"), GenericFeedback)
    assert isinstance(get_feedback_strategy("error_case"), ErrorCaseFeedback)
    assert isinstance(get_feedback_strategy("score"), ScoreFeedback)
    try:
        get_feedback_strategy("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy")


def _kb_tests():
    return [
        test_kb_one_shot_includes_example_and_arch,
        test_kb_few_shot_includes_multiple_examples,
        test_kb_zero_shot_has_no_examples,
        test_kb_backend_display_reflected,
        test_kb_rejects_unknown_gpu,
    ]


if __name__ == "__main__":
    import traceback

    names = [n for n in dir() if n.startswith("test_")]
    passed = skipped = 0
    for name in names:
        fn = globals()[name]
        if name.startswith("test_kb_") and not _kb_assets_available():
            print(f"SKIP {name} (KB assets not found)")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(names) - skipped} passed, {skipped} skipped")
    raise SystemExit(0 if passed == len(names) - skipped else 1)
