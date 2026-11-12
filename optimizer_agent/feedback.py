"""Stage-aware feedback, inspired by ``robust_kernelbench``.

``robust_kernelbench`` classifies the previous attempt (compile error, runtime
error, incorrect, correct) and composes a repair prompt. That is a good pattern
for *any* optimization task, not just kernels. This module therefore separates:

1. **Classification** -- maps an evaluator result to a fine failure stage.
2. **Generic categories** -- every fine stage also has a coarse ``failure_kind``
   (``empty`` / ``eval_error`` / ``invalid`` / ``valid``).
3. **Hint profiles** -- ``generic`` wording by default; ``kernel`` wording for
   CUDA-style tasks; or a caller-supplied dict.
4. **Metric comparison** -- configured by ``score_key`` / ``baseline_key`` /
   ``higher_is_better``, so runtime (lower is better) and accuracy (higher is
   better) both work.
5. **Strategies** -- ``none``, ``prompt``, ``rules``, ``rules_llm``, ``llm`` and
   ``evaluator``, selected per task. Custom strategies can be registered.

The evaluator can also classify itself: a ``failure_stage``, ``failure_kind``
or ``error_kind``/``error_type`` field in its JSON takes priority over all
heuristics. That is the main extension point for non-code tasks.
"""

from optimizer_agent.utils import eval_detail

# Fine stages (backwards-compatible with the kernel-oriented version).
EMPTY = "empty"
EVAL_ERROR = "eval_error"
COMPILE_ERROR = "compile_error"
RUNTIME_ERROR = "runtime_error"
INCORRECT = "incorrect"
INVALID = "invalid"
IMPROVE = "improve"

ALL_STAGES = (
    EMPTY,
    EVAL_ERROR,
    COMPILE_ERROR,
    RUNTIME_ERROR,
    INCORRECT,
    INVALID,
    IMPROVE,
)

# Coarse, domain-agnostic categories.
KIND_EMPTY = "empty"
KIND_EVAL_ERROR = "eval_error"
KIND_INVALID = "invalid"
KIND_VALID = "valid"

KIND_OF_STAGE = {
    EMPTY: KIND_EMPTY,
    EVAL_ERROR: KIND_EVAL_ERROR,
    COMPILE_ERROR: KIND_INVALID,
    RUNTIME_ERROR: KIND_INVALID,
    INCORRECT: KIND_INVALID,
    INVALID: KIND_INVALID,
    IMPROVE: KIND_VALID,
}

ERROR_KIND_OF_STAGE = {
    EMPTY: "empty",
    EVAL_ERROR: "evaluator",
    COMPILE_ERROR: "compile",
    RUNTIME_ERROR: "runtime",
    INCORRECT: "incorrect",
    INVALID: "invalid",
    IMPROVE: "",
}

_COMPILE_HINTS = (
    "compile",
    "compilation",
    "nvcc",
    "ptxas",
    "ninja",
    "undefined reference",
    "error: expected",
    "failed to build",
)
_RUNTIME_HINTS = (
    "runtime error",
    "traceback",
    "cuda error",
    "illegal memory access",
    "segmentation fault",
    "out of memory",
    "exception",
)
_INVALID_HINTS = (
    "constraint",
    "invalid",
    "wrong",
    "mismatch",
    "incorrect",
)

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
# Generic wording. ``{detail}`` already contains a leading newline when there is
# an error, and ``{metric_summary}`` is empty when no metric is available.
GENERIC_HINTS = {
    KIND_EMPTY: (
        "The previous attempt returned no artifact. Return a complete artifact "
        "in the requested format."
    ),
    KIND_EVAL_ERROR: (
        "The evaluator could not score the previous attempt. Make sure the "
        "artifact satisfies the evaluator contract.{detail}"
    ),
    KIND_INVALID: (
        "The previous attempt was not accepted ({error_kind}). Fix that before "
        "trying to optimize.{detail}"
    ),
    KIND_VALID: (
        "The previous attempt was accepted. Improve {metric_name} while "
        "preserving validity. {metric_summary}"
    ),
}

# Kernel-specific wording, matching the original feedback. Select with
# ``feedback_profile="kernel"`` (or ``"feedback_profile": "kernel"`` in a prompt
# config). The fine stages are still available to prompts as ``{failure_stage}``.
KERNEL_HINTS = {
    EMPTY: (
        "The previous attempt returned no usable code. Return the complete, "
        "runnable implementation in one fenced code block."
    ),
    EVAL_ERROR: (
        "The evaluator could not score the previous attempt. Make sure the "
        "code is self-contained, runs standalone, and follows the evaluator "
        "contract.{detail}"
    ),
    COMPILE_ERROR: (
        "The previous attempt failed to compile. Fix the compilation errors "
        "before trying to optimize.{detail}"
    ),
    RUNTIME_ERROR: (
        "The previous attempt compiled but failed while loading or running. "
        "Fix the runtime error before optimizing.{detail}"
    ),
    INCORRECT: (
        "The previous attempt ran but produced incorrect output. Restore "
        "correctness before optimizing.{detail}"
    ),
    INVALID: GENERIC_HINTS[KIND_INVALID],
    IMPROVE: (
        "The previous attempt was correct. Improve performance without "
        "changing its behaviour. {metric_summary}"
    ),
}

GENERIC_PROFILE = {
    "hints": GENERIC_HINTS,
    # None means "use the agent's selection score key".
    "score_key": None,
    "baseline_key": "baseline_score",
    "higher_is_better": True,
    "metric_name": "score",
    "use_text_heuristics": False,
}

# For kernel work the meaningful feedback metric is runtime, not the selection
# score. The search still uses the global ``score_key`` (usually combined_score).
KERNEL_PROFILE = {
    "hints": KERNEL_HINTS,
    "score_key": "runtime_custom_ms",
    "baseline_key": "runtime_original_ms",
    "higher_is_better": False,
    "metric_name": "runtime (ms)",
    "use_text_heuristics": True,
}

PROFILES = {
    "generic": GENERIC_PROFILE,
    "kernel": KERNEL_PROFILE,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "ok", "success", "passed", "valid"):
            return True
        if text in ("false", "0", "no", "n", "fail", "failed", "error", "invalid", ""):
            return False
    return None


def _flag(mapping, *keys):
    """First present boolean flag among ``keys`` (``None`` if absent)."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _as_bool(mapping[key])
    return None


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _text(eval_result):
    scores = eval_result.get("scores") or {}
    parts = [
        str(eval_result.get("stderr") or ""),
        str(eval_result.get("stdout") or ""),
    ]
    for key in ("error", "main_error", "message", "error_message", "traceback"):
        if scores.get(key):
            parts.append(str(scores[key]))
    return "\n".join(parts).lower()


def _normalise_stage(value):
    """Map evaluator-provided stage/kind/error-type strings to a fine stage."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if text in ("empty", "missing", "no_output", "no_artifact"):
        return EMPTY
    if text in ("valid", "ok", "success", "passed", "accepted", "improve", "improved"):
        return IMPROVE
    if "compile" in text:
        return COMPILE_ERROR
    if "runtime" in text or "execut" in text:
        return RUNTIME_ERROR
    if "incorrect" in text or "wrong" in text or "mismatch" in text:
        return INCORRECT
    if "constraint" in text or "invalid" in text or "infeasible" in text:
        return INVALID
    if "timeout" in text or "eval" in text or "error" in text or "exception" in text:
        return EVAL_ERROR
    return None


# Public alias for prompt/config helpers.
normalise_stage = _normalise_stage


def _explicit_stage(eval_result, scores):
    """Honour evaluator-provided classification before using heuristics."""
    for source in (scores, eval_result):
        for key in ("failure_stage", "stage", "failure_kind", "outcome"):
            stage = _normalise_stage(source.get(key))
            if stage is not None:
                return stage
        for key in ("error_kind", "error_type", "kind"):
            stage = _normalise_stage(source.get(key))
            if stage is not None and stage != IMPROVE:
                return stage
    return None


def _score_of(scores, key, fallback=True):
    value = _finite((scores or {}).get(key))
    if value is None and fallback and key != "combined_score":
        value = _finite((scores or {}).get("combined_score"))
    return value


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_result(
    eval_result,
    code="",
    score_key="combined_score",
    use_text_heuristics=False,
):
    """Return a fine failure stage for one evaluated candidate.

    Priority order:

    1. evaluator-provided ``failure_stage`` / ``failure_kind`` / ``error_kind``;
    2. generic flags (``valid``, ``success``, ``passed``, ``correct``, ...);
    3. kernel-compatible aliases (``compiled``, ``loading_success``,
       ``correctness``, ...), kept as optional signals;
    4. optional text heuristics (nvcc/traceback) for code-like evaluators;
    5. the selection score (finite => ``improve``, missing => ``eval_error``).

    ``use_text_heuristics`` is off for the generic profile so an evaluator crash
    containing the word "traceback" is not mistaken for a candidate runtime
    error. The ``kernel`` profile turns it on.
    """
    if not (code or "").strip():
        return EMPTY

    eval_result = eval_result or {}
    scores = eval_result.get("scores") or {}
    status = eval_result.get("status", "ok")

    explicit = _explicit_stage(eval_result, scores)
    if explicit is not None:
        return explicit

    # Generic flags. These are what new, non-kernel evaluators should provide.
    valid = _flag(scores, "valid", "success", "passed", "accepted", "feasible")
    if valid is False:
        return INVALID
    constraints = _flag(
        scores, "constraints_satisfied", "constraint_satisfied", "correct"
    )
    if constraints is False:
        return INVALID

    # Kernel-compatible aliases (optional; harmless for other tasks).
    if _flag(scores, "compiled", "model_new_available") is False:
        return COMPILE_ERROR
    if _flag(scores, "second_compile_success") is False:
        return COMPILE_ERROR
    if scores.get("compilation_error"):
        return COMPILE_ERROR
    if _flag(scores, "loading_success") is False:
        return RUNTIME_ERROR
    if _flag(scores, "correctness_success", "execution_success") is False:
        return RUNTIME_ERROR
    if scores.get("runtime_error") or scores.get("execution_error"):
        return RUNTIME_ERROR
    if _flag(scores, "correctness") is False:
        return INCORRECT

    if status != "ok":
        if use_text_heuristics:
            text = _text(eval_result)
            if any(hint in text for hint in _COMPILE_HINTS):
                return COMPILE_ERROR
            if any(hint in text for hint in _RUNTIME_HINTS):
                return RUNTIME_ERROR
            if any(hint in text for hint in _INVALID_HINTS):
                return INVALID
        return EVAL_ERROR

    # A successful process with no usable score is still an evaluation failure.
    if _score_of(scores, score_key) is None:
        return EVAL_ERROR
    return IMPROVE


def failure_kind(stage):
    """Coarse, domain-agnostic category for a fine stage."""
    return KIND_OF_STAGE.get(stage, KIND_INVALID)


def error_kind(stage, scores=None):
    """Fine reason for prompts: compile / runtime / incorrect / evaluator / ..."""
    explicit = (scores or {}).get("error_kind") or (scores or {}).get("error_type")
    if explicit:
        return str(explicit).strip().lower()
    return ERROR_KIND_OF_STAGE.get(stage, "invalid")


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------
def metric_change_of(
    scores,
    score_key="combined_score",
    baseline_key="baseline_score",
    higher_is_better=True,
):
    """Return ``(change, improved)`` relative to a baseline.

    ``change > 1`` always means better, regardless of whether higher or lower
    values are better. Returns ``(None, None)`` when no baseline is available.
    """
    scores = scores or {}
    score = _score_of(scores, score_key, fallback=False)
    if score is None:
        return None, None

    baseline = None
    keys = [baseline_key] if baseline_key else []
    keys += ["baseline_score", "score_baseline", "baseline"]
    for key in keys:
        if key and key in scores:
            baseline = _finite(scores[key])
            if baseline is not None:
                break
    if baseline is None or baseline == 0:
        return None, None

    if higher_is_better:
        change = score / baseline
    else:
        if score == 0:
            return None, None
        change = baseline / score
    return change, change > 1


def speedup_of(scores):
    """Backwards-compatible helper for runtime-style results."""
    return metric_change_of(
        scores,
        score_key="runtime_custom_ms",
        baseline_key="runtime_original_ms",
        higher_is_better=False,
    )[0]


def _truncate(text, limit=1200):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _profile_config(profile, score_key=None):
    if isinstance(profile, dict):
        config = dict(GENERIC_PROFILE)
        config.update(profile)
    else:
        config = dict(PROFILES.get(profile or "generic", GENERIC_PROFILE))
    if not config.get("score_key"):
        config["score_key"] = score_key or "combined_score"
    return config


def _metric_summary(scores, config, max_chars=1200):
    scores = scores or {}
    score_key = config.get("score_key", "combined_score")
    score = _score_of(scores, score_key, fallback=False)
    label = config.get("metric_name", "score")
    if score is None and score_key != "combined_score":
        score = _finite(scores.get("combined_score"))
        label = "score"
    change, improved = metric_change_of(
        scores,
        score_key=score_key,
        baseline_key=config.get("baseline_key"),
        higher_is_better=config.get("higher_is_better", True),
    )
    baseline = None
    for key in (config.get("baseline_key"), "baseline_score", "baseline"):
        if key and key in scores:
            baseline = _finite(scores[key])
            if baseline is not None:
                break

    parts = []
    if score is not None:
        parts.append(f"{label}={score:.6g}")
    if baseline is not None:
        parts.append(f"baseline={baseline:.6g}")
    if change is not None:
        word = "better" if improved else "worse"
        parts.append(f"change={change:.4f}x ({word})")
    return _truncate(", ".join(parts), max_chars)


def feedback_hint(
    stage,
    scores=None,
    error="",
    profile="generic",
    hints=None,
    max_chars=1200,
    score_key=None,
):
    """A deterministic instruction for the next attempt.

    ``profile`` is ``"generic"``, ``"kernel"``, or a dict overriding keys such as
    ``hints``, ``score_key``, ``baseline_key``, ``higher_is_better`` and
    ``metric_name``. ``hints`` is a per-stage/per-kind override mapping.
    """
    config = _profile_config(profile, score_key=score_key)
    if hints:
        merged = dict(config.get("hints") or {})
        merged.update(hints)
        config = dict(config, hints=merged)

    kind = failure_kind(stage)
    template = (config.get("hints") or {}).get(stage)
    if template is None:
        template = (config.get("hints") or {}).get(kind)
    if template is None:
        template = GENERIC_HINTS.get(kind, "")

    detail = _truncate(error, max_chars)
    context = {
        "detail": f"\n{detail}" if detail else "",
        "error": detail,
        "failure_stage": stage,
        "failure_kind": kind,
        "error_kind": error_kind(stage, scores),
        "metric_name": config.get("metric_name", "score"),
        "metric_summary": _metric_summary(scores, config, max_chars=max_chars),
    }
    try:
        return template.format(**context).strip()
    except (KeyError, IndexError, ValueError):
        return template.strip()


def apply_feedback(
    prompt_data,
    eval_result,
    code="",
    profile="generic",
    hints=None,
    score_key="combined_score",
):
    """Classify a result and merge stage-aware feedback into ``prompt_data``.

    Keys added: ``failure_stage`` (fine), ``failure_kind`` (generic),
    ``error_kind``, ``feedback``, ``previous_artifact`` (and the old
    ``previous_implementation`` alias), ``score``, ``metric_name``,
    ``metric_change`` and ``improvement`` when a metric is available.
    """
    scores = (eval_result or {}).get("scores") or {}
    config = _profile_config(profile, score_key=score_key)
    stage = classify_result(
        eval_result,
        code=code,
        score_key=score_key,
        use_text_heuristics=config.get("use_text_heuristics", False),
    )
    error = eval_detail(eval_result)
    hint = feedback_hint(
        stage,
        scores=scores,
        error=error,
        profile=profile,
        hints=hints,
        score_key=score_key,
    )

    config = _profile_config(profile, score_key=score_key)
    change, improved = metric_change_of(
        scores,
        score_key=config.get("score_key", "combined_score"),
        baseline_key=config.get("baseline_key"),
        higher_is_better=config.get("higher_is_better", True),
    )

    prompt_data["failure_stage"] = stage
    prompt_data["failure_kind"] = failure_kind(stage)
    prompt_data["error_kind"] = error_kind(stage, scores)
    prompt_data["feedback"] = hint
    prompt_data["feedback_hint"] = hint
    prompt_data["previous_artifact"] = code
    # Backwards-compatible alias for code-oriented prompt configs.
    prompt_data["previous_implementation"] = code

    score = _score_of(scores, score_key)
    if score is not None:
        prompt_data["score"] = score
    metric_value = _score_of(scores, config.get("score_key", score_key), fallback=False)
    if metric_value is not None:
        prompt_data["metric_value"] = metric_value
    prompt_data["metric_name"] = config.get("metric_name", "score")
    if change is not None:
        prompt_data["metric_change"] = change
        prompt_data["improvement"] = improved
        # Keep the old name for runtime-oriented prompts.
        if not config.get("higher_is_better", True):
            prompt_data["speedup"] = change
    for key in ("runtime", "runtime_original", "runtime_custom_ms", "runtime_original_ms"):
        if key in scores:
            prompt_data[key] = scores[key]

    if error and stage != IMPROVE:
        prompt_data["error"] = error
    else:
        # Do not carry benign stderr warnings from a successful run forward.
        prompt_data.pop("error", None)

    return stage


# ---------------------------------------------------------------------------
# Feedback strategies
# ---------------------------------------------------------------------------
def has_error_signal(eval_result):
    """True when a result should surface an error to the next prompt."""
    if not eval_result:
        return False
    scores = eval_result.get("scores") or {}
    if eval_result.get("status", "ok") != "ok":
        return True
    for key in ("error", "message", "error_message", "traceback", "main_error"):
        if scores.get(key):
            return True
    for key in (
        "valid",
        "success",
        "passed",
        "accepted",
        "feasible",
        "correct",
        "correctness",
        "compiled",
        "loading_success",
        "correctness_success",
        "execution_success",
        "constraints_satisfied",
    ):
        if key in scores and _as_bool(scores[key]) is False:
            return True
    return False


def _resolve_profile(profile, metric=None):
    """Resolve a profile name/dict and merge an optional ``metric`` block."""
    if isinstance(profile, dict):
        resolved = dict(profile)
    else:
        resolved = dict(PROFILES.get(profile or "generic", GENERIC_PROFILE))
    if isinstance(metric, dict):
        resolved.update(metric)
    return resolved


class FeedbackStrategy:
    """Base class for a feedback strategy.

    A strategy mutates ``prompt_data`` after evaluation. It may also declare
    that the agent should run the LLM reviewer:

    - ``llm_mode = None``    -- deterministic only
    - ``llm_mode = "augment"`` -- append the LLM review to the deterministic hint
    - ``llm_mode = "replace"`` -- use the LLM review instead of the hint
    """

    name = "base"
    llm_mode = None
    has_hint = False

    def __init__(self, config=None):
        self.config = dict(config or {})

    @property
    def wants_llm(self):
        return self.llm_mode is not None

    def apply(self, prompt_data, eval_result, artifact="", score_key="combined_score"):
        # Every strategy keeps the previous artifact available to prompts.
        prompt_data["previous_artifact"] = artifact
        prompt_data["previous_implementation"] = artifact
        return {}


class NoFeedbackStrategy(FeedbackStrategy):
    """Do nothing beyond the raw ``result``/``scores`` injected by the agent."""

    name = "none"


class PromptFeedbackStrategy(FeedbackStrategy):
    """No structured hint; prompt config does all the work."""

    name = "prompt"


class RulesFeedbackStrategy(FeedbackStrategy):
    """Classify the result and inject a deterministic repair hint."""

    name = "rules"
    has_hint = True

    def __init__(self, config=None):
        super().__init__(config)
        self.profile = _resolve_profile(
            self.config.get("profile", "generic"), self.config.get("metric")
        )
        self.hints = self.config.get("hints")

    def apply(self, prompt_data, eval_result, artifact="", score_key="combined_score"):
        super().apply(prompt_data, eval_result, artifact, score_key=score_key)
        apply_feedback(
            prompt_data,
            eval_result,
            artifact,
            profile=self.profile,
            hints=self.hints,
            score_key=score_key,
        )
        return {}


class RulesLLMFeedbackStrategy(RulesFeedbackStrategy):
    """Classify + hint, then add an LLM reviewer."""

    name = "rules_llm"
    llm_mode = "augment"


class LLMFeedbackStrategy(FeedbackStrategy):
    """No deterministic hint; an LLM reviewer sees the raw result."""

    name = "llm"
    llm_mode = "replace"


class EvaluatorFeedbackStrategy(FeedbackStrategy):
    """Use feedback supplied by the task-specific evaluator.

    The evaluator can return ``feedback`` / ``hint`` / ``feedback_hint`` plus an
    optional ``failure_stage``, ``failure_kind`` or ``error_kind``. This is the
    escape hatch for domains whose failure taxonomy only the evaluator knows.
    """

    name = "evaluator"
    has_hint = True

    def apply(self, prompt_data, eval_result, artifact="", score_key="combined_score"):
        super().apply(prompt_data, eval_result, artifact, score_key=score_key)
        scores = (eval_result or {}).get("scores") or {}
        text = (
            scores.get("feedback")
            or scores.get("hint")
            or scores.get("feedback_hint")
            or scores.get("critique")
            or ""
        )
        if text:
            prompt_data["feedback"] = str(text)
            prompt_data["feedback_hint"] = str(text)

        stage = _normalise_stage(
            scores.get("failure_stage")
            or scores.get("failure_kind")
            or scores.get("error_kind")
            or scores.get("error_type")
        )
        if stage is not None:
            prompt_data["failure_stage"] = stage
            prompt_data["failure_kind"] = failure_kind(stage)
            prompt_data["error_kind"] = error_kind(stage, scores)
        return {}


_FEEDBACK_STRATEGY_CLASSES = {
    "none": NoFeedbackStrategy,
    "prompt": PromptFeedbackStrategy,
    "rules": RulesFeedbackStrategy,
    "rules_llm": RulesLLMFeedbackStrategy,
    "llm": LLMFeedbackStrategy,
    "evaluator": EvaluatorFeedbackStrategy,
}

# Friendly aliases are accepted in config files as well as class names.
FEEDBACK_STRATEGY_ALIASES = {
    "off": "none",
    "prompt_only": "prompt",
    "raw": "prompt",
    "classify_hint": "rules",
    "classify": "rules",
    "hybrid": "rules_llm",
    "classify_hint_llm": "rules_llm",
    "critic": "llm",
    "evaluator_feedback": "evaluator",
}

FEEDBACK_STRATEGY_NAMES = tuple(_FEEDBACK_STRATEGY_CLASSES)


def build_feedback_strategy(config=None, name=None):
    """Instantiate a feedback strategy from a name/config dict.

    ``config`` may be a strategy name or contain ``strategy`` (or ``name``),
    ``profile``, ``hints``, and a nested ``metric`` block (``score_key``,
    ``baseline_key``, ``higher_is_better``, ``metric_name``).
    """
    if isinstance(config, str):
        config = {"strategy": config}
    config = dict(config or {})
    strategy_name = (
        name
        or config.pop("strategy", None)
        or config.pop("name", None)
        or "rules"
    )
    strategy_name = FEEDBACK_STRATEGY_ALIASES.get(strategy_name, strategy_name)
    try:
        strategy_class = _FEEDBACK_STRATEGY_CLASSES[strategy_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feedback strategy {strategy_name!r}. "
            f"Available: {sorted(FEEDBACK_STRATEGY_NAMES)}"
        ) from exc
    return strategy_class(config)


def register_feedback_strategy(name, strategy_class, aliases=()):
    """Register a custom strategy class (e.g. a future reflection/memory one)."""
    global FEEDBACK_STRATEGY_NAMES
    _FEEDBACK_STRATEGY_CLASSES[name] = strategy_class
    for alias in aliases:
        FEEDBACK_STRATEGY_ALIASES[alias] = name
    FEEDBACK_STRATEGY_NAMES = tuple(_FEEDBACK_STRATEGY_CLASSES)
    return strategy_class
