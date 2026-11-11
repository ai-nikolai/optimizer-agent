"""The search loop.

``Agent`` owns control flow only: pick predecessors, build prompts, call the
model, evaluate the candidate in a subprocess, feed the result back, optionally
debug, and record artifacts. It deliberately has no knowledge of CUDA or of
what the score means.
"""

import json
import logging
import shutil
import time
import traceback
from pathlib import Path

from optimizer_agent.candidates import (
    CandidatesBestNAll,
    CandidatesBestNLast,
    MapElitesCandidates,
    SingleCandidate,
)
from optimizer_agent.diffs import DiffApplyError, apply_diff
from optimizer_agent.feedback import (
    build_feedback_strategy,
    has_error_signal,
)
from optimizer_agent.llm import FakeLLMServer, LLMError, LLMServer
from optimizer_agent.prompts import (
    DEFAULT_PROMPT_CONFIG,
    create_prompt,
    resolve_prompt_config,
    resolve_stage_configs,
)
from optimizer_agent.utils import (
    DEFAULT_EVAL_TIMEOUT,
    eval_detail,
    extract_response,
    rank_score_of,
    read_file,
    run_evaluation,
    score_of,
    summarize_result,
    write_file,
)

_SELECTION = {
    "bestN_all": CandidatesBestNAll,
    "bestN_last": CandidatesBestNLast,
    "map_elites": MapElitesCandidates,
}

# Results of the previous attempt must not leak into the next one if the
# selected strategy does not set them (e.g. evaluator stops supplying feedback).
_FEEDBACK_KEYS = (
    "feedback",
    "feedback_hint",
    "llm_feedback",
    "failure_stage",
    "failure_kind",
    "error_kind",
    "metric_name",
    "metric_value",
    "metric_change",
    "improvement",
    "speedup",
)


class Agent:
    def __init__(
        self,
        num_iterations=None,
        num_debug_iterations=None,
        num_samples=None,
        prompt_config_file=None,
        prompt_config=None,
        predecessor_selection_method=None,
        num_map_elites=None,
        max_candidates=None,
        minimize=None,
        diff_based=None,
        diff_mode=None,
        map_elites_cluster_metric=None,
        map_elites_taxonomy_file=None,
        map_elites_cluster_threshold=None,
        max_map_elites_cells=None,
        embedding_url=None,
        embedding_model=None,
        embedding_api_key=None,
        llm_timeout=None,
        llm_retries=None,
        llm_retry_backoff=None,
        resume=None,
        overwrite=None,
        use_strategy=None,
        use_feedback=None,
        use_debug=None,
        use_debug_strategy=None,
        use_debug_feedback=None,
        model_name="default",
        model_url="http://0.0.0.0:30000",
        model_api_key="<no_key>",
        model_params=None,
        eval_timeout=None,
        test=False,
        server=None,
        # Generic feedback / artifact controls (usually set in the prompt config).
        feedback_strategy=None,
        feedback_config_file=None,
        feedback_profile=None,
        feedback_hints=None,
        response_format=None,
        artifact_extension=None,
        score_key=None,
        task_description=None,
        # Legacy flat key; the nested map_elites config is preferred.
        map_elites_method=None,
    ):
        # Load config first: a task config may provide search defaults too.
        if prompt_config_file:
            with open(prompt_config_file, "r", encoding="utf-8") as handle:
                prompt_config = json.load(handle)
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG
        cfg = self.prompt_config if isinstance(self.prompt_config, dict) else {}
        if cfg.get("num_islands") not in (None, 1):
            print(
                "Warning: num_islands was removed; running a single population. "
                "Use separate experiments/processes for independent runs."
            )

        self.num_iterations = int(
            self._pick(num_iterations, cfg, "num_iterations", 1)
        )
        self.num_debug_iterations = int(
            self._pick(num_debug_iterations, cfg, "num_debug_iterations", 0)
        )
        self.num_samples = int(self._pick(num_samples, cfg, "num_samples", 1))
        self.num_map_elites = int(
            self._pick(num_map_elites, cfg, "num_map_elites", 1)
        )

        selection = (
            predecessor_selection_method
            or cfg.get("predecessor_selection_method")
            or "bestN_last"
        )
        if selection not in _SELECTION:
            raise NotImplementedError(
                f"Predecessor selection {selection!r} not allowed. "
                f"Available: {sorted(_SELECTION)}"
            )
        self.predecessor_selection_method = selection
        self.population_class = _SELECTION[selection]
        self.max_candidates = max(
            1, int(self._pick(max_candidates, cfg, "max_candidates", 20))
        )

        self.minimize = (
            bool(minimize) if minimize is not None else bool(cfg.get("minimize", False))
        )
        self.diff_based = (
            bool(diff_based)
            if diff_based is not None
            else bool(cfg.get("diff_based", False))
        )
        self.diff_mode = (diff_mode or cfg.get("diff_mode") or "unified-diff").lower()
        if self.diff_mode not in ("unified-diff", "json"):
            raise ValueError("diff_mode must be 'unified-diff' or 'json'")

        self.use_strategy = (
            bool(use_strategy)
            if use_strategy is not None
            else bool(cfg.get("use_strategy", False))
        )
        self.use_feedback = (
            bool(use_feedback)
            if use_feedback is not None
            else bool(cfg.get("use_feedback", False))
        )
        self.use_debug = (
            bool(use_debug)
            if use_debug is not None
            else bool(cfg.get("use_debug", False))
        )
        self.use_debug_strategy = (
            bool(use_debug_strategy)
            if use_debug_strategy is not None
            else bool(cfg.get("use_debug_strategy", False))
        )
        self.use_debug_feedback = (
            bool(use_debug_feedback)
            if use_debug_feedback is not None
            else bool(cfg.get("use_debug_feedback", False))
        )

        # Map-elites configuration: nested block preferred, flat keys accepted.
        map_cfg = dict(cfg.get("map_elites") or {})
        flat_map_keys = {
            "map_elites_method": "method",
            "map_elites_cluster_metric": "cluster_metric",
            "map_elites_taxonomy": "taxonomy",
            "map_elites_cluster_threshold": "cluster_threshold",
            "max_map_elites_cells": "max_cells",
            "embedding_model": "embedding_model",
            "embedding_url": "embedding_url",
            "embedding_api_key": "embedding_api_key",
        }
        for flat_key, nested_key in flat_map_keys.items():
            if nested_key not in map_cfg and cfg.get(flat_key) is not None:
                map_cfg[nested_key] = cfg[flat_key]
        if map_elites_method:
            map_cfg["method"] = map_elites_method
        if map_elites_cluster_metric:
            map_cfg["cluster_metric"] = map_elites_cluster_metric
        if map_elites_cluster_threshold is not None:
            map_cfg["cluster_threshold"] = map_elites_cluster_threshold
        if max_map_elites_cells is not None:
            map_cfg["max_cells"] = max_map_elites_cells
        if embedding_url:
            map_cfg["embedding_url"] = embedding_url
        if embedding_model:
            map_cfg["embedding_model"] = embedding_model
        if embedding_api_key:
            map_cfg["embedding_api_key"] = embedding_api_key
        if map_elites_taxonomy_file:
            with open(map_elites_taxonomy_file, "r", encoding="utf-8") as handle:
                taxonomy = json.load(handle)
            if not isinstance(taxonomy, dict):
                raise ValueError("map_elites taxonomy must be a JSON object")
            map_cfg["taxonomy"] = taxonomy
        self.map_elites_config = map_cfg
        self.map_elites_method = (map_cfg.get("method") or "cluster").lower()
        self.map_elites_cluster_metric = (
            map_cfg.get("cluster_metric") or "jaccard"
        ).lower()
        self.map_elites_taxonomy = dict(map_cfg.get("taxonomy") or {})
        self.map_elites_cluster_threshold = float(
            map_cfg.get("cluster_threshold", 0.2)
        )
        self.max_map_elites_cells = max(1, int(map_cfg.get("max_cells", 100)))
        self.embedding_url = map_cfg.get("embedding_url")
        self.embedding_model = map_cfg.get("embedding_model")
        self.embedding_api_key = map_cfg.get("embedding_api_key")

        if self.map_elites_method not in ("cluster", "fixed", "dynamic"):
            raise ValueError(
                "map_elites.method must be cluster, fixed or dynamic"
            )
        if self.map_elites_cluster_metric not in ("jaccard", "embedding"):
            raise ValueError("map_elites.cluster_metric must be jaccard or embedding")
        if (
            self.predecessor_selection_method == "map_elites"
            and self.map_elites_method == "cluster"
            and self.map_elites_cluster_metric == "embedding"
            and (not self.embedding_url or not self.embedding_model)
        ):
            raise ValueError(
                "cluster-metric=embedding requires embedding_url and embedding_model"
            )

        self.llm_timeout = float(
            self._pick(llm_timeout, cfg, "llm_timeout", 600)
        )
        self.llm_retries = max(1, int(self._pick(llm_retries, cfg, "llm_retries", 3)))
        self.llm_retry_backoff = max(
            0.0, float(self._pick(llm_retry_backoff, cfg, "llm_retry_backoff", 1.0))
        )
        self.resume = (
            bool(resume) if resume is not None else bool(cfg.get("resume", True))
        )
        self.overwrite = (
            bool(overwrite)
            if overwrite is not None
            else bool(cfg.get("overwrite", False))
        )

        self.model_name = model_name
        self.model_url = model_url
        self.model_api_key = model_api_key
        self.model_params = model_params or {}
        self.eval_timeout = int(
            eval_timeout or cfg.get("eval_timeout") or DEFAULT_EVAL_TIMEOUT
        )

        self.improve_config = resolve_prompt_config(self.prompt_config, "improve")
        self.debug_config = resolve_prompt_config(self.prompt_config, "debug")
        self.classify_config = resolve_prompt_config(self.prompt_config, "classify")
        self.stage_configs = resolve_stage_configs(self.prompt_config, "improve")
        self.debug_stage_configs = resolve_stage_configs(self.prompt_config, "debug")

        # Let a prompt config carry the task/feedback setup so new domains can be
        # added without touching the CLI.
        self.feedback_config = self._resolve_feedback_config(
            cfg,
            feedback_config_file=feedback_config_file,
            feedback_strategy=feedback_strategy,
            feedback_profile=feedback_profile,
            feedback_hints=feedback_hints,
        )
        self.feedback_strategy = build_feedback_strategy(self.feedback_config)
        if self.use_feedback and self.feedback_strategy.llm_mode is None:
            # Backwards compatibility: --use-feedback adds an LLM reviewer.
            self.feedback_strategy.llm_mode = (
                "augment" if self.feedback_strategy.has_hint else "replace"
            )
        if self.minimize and isinstance(getattr(self.feedback_strategy, "profile", None), dict):
            self.feedback_strategy.profile["higher_is_better"] = False

        self.response_format = (
            response_format or cfg.get("response_format") or "last_code"
        ).lower()
        valid_formats = {
            "last_code",
            "last",
            "largest",
            "code",
            "python",
            "raw",
            "text",
            "none",
            "json",
        }
        if self.response_format not in valid_formats:
            raise ValueError(f"Unknown response_format {self.response_format!r}")
        self.artifact_extension = (
            artifact_extension or cfg.get("artifact_extension") or "py"
        ).lstrip(".")
        if not self.artifact_extension.isalnum():
            raise ValueError("artifact_extension must be alphanumeric")
        self.score_key = score_key or cfg.get("score_key") or "combined_score"
        self.task_description = task_description or cfg.get("task_description")

        if server is not None:
            self.server = server
        elif test:
            self.server = FakeLLMServer()
        else:
            self.server = LLMServer(
                model_url,
                model_api_key,
                timeout=self.llm_timeout,
                retries=self.llm_retries,
                retry_backoff=self.llm_retry_backoff,
            )

        if (
            self.predecessor_selection_method == "map_elites"
            and self.map_elites_method == "fixed"
            and not self.map_elites_taxonomy
        ):
            raise ValueError("fixed map-elites requires a non-empty taxonomy")

        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.num_debug_iterations < 0:
            raise ValueError("num_debug_iterations must be >= 0")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.num_map_elites < 1:
            raise ValueError("num_map_elites must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.eval_timeout < 1:
            raise ValueError("eval_timeout must be >= 1")
        if self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be > 0")
        if not 0.0 <= self.map_elites_cluster_threshold <= 1.0:
            raise ValueError("map_elites cluster_threshold must be between 0 and 1")

        self.population_kwargs = {
            "num_map_elites": self.num_map_elites,
            "max_candidates": self.max_candidates,
        }
        if self.predecessor_selection_method == "map_elites":
            self.population_kwargs.update(
                {
                    "method": self.map_elites_method,
                    "cluster_metric": self.map_elites_cluster_metric,
                    "taxonomy": self.map_elites_taxonomy,
                    "classify": self.classify_artifact,
                    "embed": self.embed_artifact,
                    "max_cells": self.max_map_elites_cells,
                    "cluster_threshold": self.map_elites_cluster_threshold,
                }
            )

        self.experiments_dir = None
        self.baseline_path = None
        self.evaluate_path = None
        self.reference_path = None
        self.population = None
        self.history = []
        self.summary = None
        self.logger = None
        self._events_path = None
        self._resumed = False

    @staticmethod
    def _pick(value, cfg, key, default):
        """Explicit argument > task config > default."""
        if value is not None:
            return value
        configured = cfg.get(key)
        return default if configured is None else configured

    @staticmethod
    def _resolve_feedback_config(
        prompt_config,
        feedback_config_file=None,
        feedback_strategy=None,
        feedback_profile=None,
        feedback_hints=None,
    ):
        """Merge feedback settings from the prompt config, a feedback file, CLI.

        Precedence: prompt-config defaults < dedicated feedback file < explicit
        arguments. A strategy can be a name or a dict with ``profile``/``hints``/
        ``metric`` options.
        """
        merged = {}
        config = prompt_config if isinstance(prompt_config, dict) else {}

        if config.get("feedback_profile"):
            merged["profile"] = config["feedback_profile"]
        if config.get("feedback_hints"):
            merged["hints"] = dict(config["feedback_hints"])

        strategy = config.get("feedback_strategy")
        if isinstance(strategy, str):
            merged["strategy"] = strategy
        elif isinstance(strategy, dict):
            nested = dict(strategy)
            nested_name = nested.pop("strategy", nested.pop("name", None))
            if nested_name:
                merged["strategy"] = nested_name
            merged.update(nested)

        if feedback_config_file:
            with open(feedback_config_file, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("feedback config must be a JSON object")
            loaded = dict(loaded)
            loaded_name = loaded.pop("strategy", loaded.pop("name", None))
            if loaded_name:
                merged["strategy"] = loaded_name
            if "hints" in loaded:
                merged["hints"] = {**merged.get("hints", {}), **loaded.pop("hints")}
            merged.update(loaded)

        if feedback_strategy:
            merged["strategy"] = feedback_strategy
        if feedback_profile:
            merged["profile"] = feedback_profile
        if feedback_hints:
            merged["hints"] = {**merged.get("hints", {}), **feedback_hints}

        return merged

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # experiment setup / logging / resume
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_rmtree(path):
        resolved = Path(path).resolve()
        if resolved in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
            raise ValueError(f"Refusing to delete unsafe experiment path: {resolved}")
        if resolved.parent == resolved:
            raise ValueError(f"Refusing to delete filesystem root: {resolved}")
        if len(resolved.parts) <= 2:
            raise ValueError(
                f"Refusing to delete a top-level directory as an experiment: {resolved}"
            )
        shutil.rmtree(resolved)

    def _prepare_experiment_dir(self):
        """Create the experiment dir, resuming a completed run or starting fresh."""
        path = self.experiments_dir
        if self.overwrite and path.exists():
            self._safe_rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        summary_path = path / "summary.json"
        if self.resume and not self.overwrite and summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as handle:
                    loaded_summary = json.load(handle)
                if loaded_summary.get("completed", True):
                    self.summary = loaded_summary
                    self.history = self.summary.get("history", [])
                    self._resumed = True
                    print(f"Resuming completed experiment: {path}")
                    return True
                print("Warning: partial summary found; restarting this run.")
            except (OSError, ValueError):
                print("Warning: existing summary.json is unreadable; restarting run")
        elif summary_path.exists() and not self.resume:
            print("Existing completed summary found; rerunning (resume disabled).")
        elif any(path.iterdir()):
            print(
                "Warning: experiment directory is non-empty without a completed "
                "summary; restarting this run from scratch."
            )
        return False

    def _setup_logging(self):
        self.logger = logging.getLogger(f"optimizer_agent.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(self.experiments_dir / "run.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream)
        self.logger.addHandler(file_handler)
        self._events_path = self.experiments_dir / "events.jsonl"

    def _close_logging(self):
        if not self.logger:
            return
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            finally:
                self.logger.removeHandler(handler)

    def _event(self, event, **fields):
        if not self._events_path:
            return
        payload = {"ts": time.time(), "event": event}
        payload.update(fields)
        try:
            with open(self._events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except OSError:
            pass

    def _write_config_snapshot(self):
        snapshot = {
            "prompt_config": self.prompt_config,
            "feedback_config": self.feedback_config,
            "search": {
                "num_iterations": self.num_iterations,
                "num_debug_iterations": self.num_debug_iterations,
                "num_samples": self.num_samples,
                "predecessor_selection_method": self.predecessor_selection_method,
                "num_map_elites": self.num_map_elites,
                "max_candidates": self.max_candidates,
                "minimize": self.minimize,
                "resume": self.resume,
                "overwrite": self.overwrite,
            },
            "map_elites": {
                "method": self.map_elites_method,
                "cluster_metric": self.map_elites_cluster_metric,
                "taxonomy": self.map_elites_taxonomy,
                "cluster_threshold": self.map_elites_cluster_threshold,
                "max_cells": self.max_map_elites_cells,
                "embedding_model": self.embedding_model,
                "embedding_url": self.embedding_url,
            },
            "artifact": {
                "response_format": self.response_format,
                "artifact_extension": self.artifact_extension,
                "score_key": self.score_key,
                "diff_based": self.diff_based,
                "diff_mode": self.diff_mode,
                "task_description": self.task_description,
            },
            "model": {
                "name": self.model_name,
                "url": self.model_url,
                "params": self.model_params,
                "llm_timeout": self.llm_timeout,
                "llm_retries": self.llm_retries,
                "llm_retry_backoff": self.llm_retry_backoff,
            },
        }
        write_file(
            self.experiments_dir,
            "config.json",
            json.dumps(snapshot, indent=2, default=str),
        )

    def _validate_inputs(self):
        for label, path in (
            ("baseline", self.baseline_path),
            ("evaluate", self.evaluate_path),
        ):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"{label} file not found: {path!r}")
        if self.reference_path and not Path(self.reference_path).is_file():
            raise FileNotFoundError(
                f"reference file not found: {self.reference_path!r}"
            )

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(self, baseline, evaluate, reference=None, experiments_dir="opt_agent_experiment"):
        self.experiments_dir = Path(experiments_dir)
        self.baseline_path = baseline
        self.evaluate_path = evaluate
        self.reference_path = reference
        self.population = None
        self.history = []
        self._validate_inputs()

        if self._prepare_experiment_dir():
            return self.history

        self._setup_logging()
        write_file(self.experiments_dir, "metadata.txt", str(self))
        self._write_config_snapshot()
        self._event(
            "run_start",
            baseline=str(baseline),
            evaluate=str(evaluate),
            reference=str(reference) if reference else None,
            selection=self.predecessor_selection_method,
            feedback=self.feedback_strategy.name,
        )
        self.logger.info(
            "run_start selection=%s feedback=%s minimize=%s",
            self.predecessor_selection_method,
            self.feedback_strategy.name,
            self.minimize,
        )

        try:
            for iteration in range(self.num_iterations):
                print("=" * 36)
                print(f"---- Running Iteration: {iteration}/{self.num_iterations} ----")
                print(self)
                self.logger.info(
                    "iteration %s/%s", iteration + 1, self.num_iterations
                )
                self._event("iteration_start", iteration=iteration)
                self.run_single_iteration(iteration)
                self._event("iteration_end", iteration=iteration)
            self._save_summary()
        except KeyboardInterrupt:
            self.logger.warning("interrupted; writing partial summary")
            self._event("run_interrupted")
            self._save_summary(completed=False)
            raise
        except Exception as exc:  # noqa: BLE001 - persist diagnostics then re-raise
            self.logger.exception("run failed: %s", exc)
            self._event("run_error", error=traceback.format_exc())
            self._save_summary(completed=False)
            raise
        finally:
            self._event("run_end")
            self._close_logging()
        return self.history

    # ------------------------------------------------------------------
    # iteration control flow
    # ------------------------------------------------------------------
    def run_single_iteration(self, iteration_idx):
        if iteration_idx == 0:
            population = self.population_class(**self.population_kwargs)
            self.population = population
            num_generations = (
                self.num_map_elites
                if self.predecessor_selection_method == "map_elites"
                else self.num_samples
            )

            for sample_idx in range(num_generations):
                prompt_data = self.get_prompt_data()
                prompt_data["current_implementation"] = read_file(self.baseline_path)
                prompt_data["current_artifact"] = prompt_data["current_implementation"]
                candidate = self.run_single_generation(
                    iteration_idx, prompt_data, additional_index=sample_idx
                )
                population.update_candidates(candidate, current_iteration=iteration_idx)
                self.log_candidates(
                    iteration_idx, population, additional_index=sample_idx
                )
            return

        population = self.population
        print(population)
        num_generations = (
            1 if self.predecessor_selection_method == "map_elites" else self.num_samples
        )

        for best_idx, (key, parent) in enumerate(population.get_best().items()):
            for sample_idx in range(num_generations):
                additional_index = best_idx * num_generations + sample_idx
                prompt_data = dict(parent.prompt_data)
                # This is the important part: the next generation starts from
                # the parent's actual artifact, not from the original baseline.
                try:
                    prompt_data["current_implementation"] = read_file(parent.path)
                    prompt_data["current_artifact"] = prompt_data["current_implementation"]
                except OSError:
                    prompt_data.setdefault(
                        "current_implementation", parent.prompt_data.get("new_implementation", "")
                    )
                    prompt_data.setdefault(
                        "current_artifact", prompt_data["current_implementation"]
                    )

                candidate = self.run_single_generation(
                    iteration_idx,
                    prompt_data,
                    other=key,
                    additional_index=additional_index,
                )
                population.update_candidates(candidate, current_iteration=iteration_idx)
                self.log_candidates(
                    iteration_idx, population, additional_index=additional_index
                )

    # ------------------------------------------------------------------
    # one generation (+ optional debug attempts)
    # ------------------------------------------------------------------
    def run_single_generation(
        self, iteration_idx, prompt_data, other=None, additional_index=None
    ):
        # A strategy belongs to one attempt only; feedback/results are inherited.
        prompt_data.pop("strategy", None)

        selected_config = self.select_prompt_config(
            prompt_data, self.improve_config, self.stage_configs
        )
        response_data, prompt_data = self.generate_response(
            prompt_data, selected_config, self.use_strategy
        )
        solutions_dir, prompt_data, target_path, eval_result = self.eval_and_save(
            iteration_idx, response_data, prompt_data, additional_index
        )
        response_data.update(
            self.run_feedback(
                prompt_data,
                eval_result,
                selected_config,
                self.feedback_strategy.wants_llm,
            )
        )
        self.save_response_data(solutions_dir, response_data)

        # Debug loop: regenerate from the failed artifact until it is accepted.
        if self._is_failed(eval_result, prompt_data) and self.use_debug:
            attempts = self.num_debug_iterations or 1
            for _ in range(attempts):
                debug_data = dict(prompt_data)
                debug_data["current_implementation"] = (
                    response_data.get("core_response")
                    or prompt_data.get("current_implementation", "")
                )
                debug_data["current_artifact"] = debug_data["current_implementation"]
                debug_data["error"] = eval_detail(eval_result) or summarize_result(
                    eval_result
                )
                debug_data["result"] = summarize_result(eval_result)
                debug_data.pop("strategy", None)

                debug_response, debug_data = self.generate_response(
                    debug_data,
                    self.select_prompt_config(
                        debug_data, self.debug_config, self.debug_stage_configs
                    ),
                    self.use_debug_strategy,
                )
                response_data.update(debug_response)
                solutions_dir, debug_data, target_path, eval_result = self.eval_and_save(
                    iteration_idx,
                    response_data,
                    debug_data,
                    additional_index,
                    is_debug=True,
                )
                prompt_data = debug_data
                response_data.update(
                    self.run_feedback(
                        prompt_data,
                        eval_result,
                        self.select_prompt_config(
                            prompt_data, self.debug_config, self.debug_stage_configs
                        ),
                        self.use_debug_feedback
                        or self.feedback_strategy.wants_llm,
                    )
                )
                self.save_response_data(solutions_dir, response_data, is_debug=True)
                if not self._is_failed(eval_result, prompt_data):
                    break

        other_vector = None
        if self.predecessor_selection_method == "map_elites":
            new_code = response_data.get("core_response", "")
            if not other:
                other = new_code
            other, _ = self.population.get_key(new_code, other)
            other_vector = getattr(self.population, "last_key_vector", None)

        raw_score = score_of(eval_result["scores"], key=self.score_key)
        rank_score = rank_score_of(
            eval_result["scores"], key=self.score_key, minimize=self.minimize
        )
        candidate = SingleCandidate(
            target_path,
            eval_result["scores"],
            other=other,
            prompt_data=prompt_data,
            score_key=self.score_key,
            minimize=self.minimize,
        )
        candidate.other_vector = other_vector
        self.history.append(
            {
                "iteration": iteration_idx,
                "sample": additional_index,
                "path": str(target_path),
                "score": raw_score,
                "rank_score": rank_score,
                "status": eval_result["status"],
                "failure_stage": prompt_data.get("failure_stage"),
                "failure_kind": prompt_data.get("failure_kind"),
                "scores": eval_result["scores"],
            }
        )
        self._event(
            "candidate",
            iteration=iteration_idx,
            sample=additional_index,
            path=str(target_path),
            score=raw_score,
            rank_score=rank_score,
            status=eval_result["status"],
            failure_stage=prompt_data.get("failure_stage"),
            failure_kind=prompt_data.get("failure_kind"),
        )
        if self.logger:
            self.logger.info(
                "candidate iteration=%s sample=%s score=%s rank=%s status=%s stage=%s",
                iteration_idx,
                additional_index,
                raw_score,
                rank_score,
                eval_result["status"],
                prompt_data.get("failure_stage"),
            )
        return candidate

    @staticmethod
    def _is_failed(eval_result, prompt_data):
        """Evaluator failure, or a strategy classified the artifact as invalid."""
        return (
            eval_result.get("status") != "ok"
            or prompt_data.get("failure_kind") == "invalid"
        )

    @staticmethod
    def select_prompt_config(prompt_data, default, overrides):
        """Pick a stage-specific prompt bundle based on the parent's outcome."""
        stage = prompt_data.get("failure_stage")
        return overrides.get(stage, default)

    def generate_response(self, prompt_data, config, use_strategy):
        """Run strategy (optional) then the core artifact-generation stage."""
        response_data = {}

        if use_strategy and "strategy" in config:
            messages = create_prompt("strategy", prompt_data, config)
            strategy = self.call_llm(messages)
            response_data["strategy_prompt"] = messages
            response_data["strategy_response_raw"] = strategy
            prompt_data["strategy"] = strategy

        messages = create_prompt("core", prompt_data, config)
        if self.diff_based:
            messages.append({"role": "user", "content": self._diff_instruction()})

        raw = self.call_llm(messages)
        diff_applied = False
        if self.diff_based:
            base_artifact = prompt_data.get("current_artifact", "")
            try:
                artifact = apply_diff(
                    base_artifact,
                    raw,
                    mode=self.diff_mode,
                    extension=self.artifact_extension,
                )
                diff_applied = True
            except DiffApplyError as exc:
                message = f"diff apply failed ({self.diff_mode}): {exc}"
                print(f"Warning: {message}; falling back to full artifact")
                if self.logger:
                    self.logger.warning(message)
                self._event("diff_fallback", mode=self.diff_mode, error=str(exc))
                artifact = extract_response(
                    raw, response_format=self.response_format
                )
        else:
            artifact = extract_response(raw, response_format=self.response_format)

        response_data["core_prompt"] = messages
        response_data["core_response_raw"] = raw
        response_data["core_response"] = artifact
        response_data["diff_applied"] = diff_applied
        prompt_data["new_implementation"] = artifact
        prompt_data["new_artifact"] = artifact
        return response_data, prompt_data

    def _diff_instruction(self):
        if self.diff_mode == "json":
            return (
                "Return ONLY a JSON object with the form "
                '{"edits": [{"old": "...", "new": "..."}, ...]}. '
                "Each old block must appear verbatim in the current artifact. "
                "Apply edits in order. Do not return the full artifact."
            )
        return (
            "Return ONLY a unified diff against the current artifact. Use standard "
            "--- / +++ / @@ headers and the file name `artifact`. Do not include "
            "the full artifact or any explanation."
        )

    def run_feedback(self, prompt_data, eval_result, config, enabled):
        """Run the optional LLM feedback stage after evaluation."""
        if not enabled:
            return {}
        if "feedback" not in config:
            print("Warning: no 'feedback' stage in prompt config; skipping.")
            return {}
        messages = create_prompt("feedback", prompt_data, config)
        llm_feedback = self.call_llm(messages)
        hint = prompt_data.get("feedback_hint") or prompt_data.get("feedback") or ""
        mode = self.feedback_strategy.llm_mode or "replace"

        if not llm_feedback.strip():
            final = hint
        elif mode == "augment":
            final = f"{hint}\n\nAdditional reviewer feedback:\n{llm_feedback}" if hint else llm_feedback
        else:
            final = llm_feedback

        if final.strip():
            prompt_data["feedback"] = final
        prompt_data["llm_feedback"] = llm_feedback
        return {
            "feedback_prompt": messages,
            "feedback_response_raw": llm_feedback,
        }

    def call_llm(self, messages):
        params = dict(self.model_params)
        params.setdefault("model", self.model_name)
        contents, _ = self.server.call_server(messages, **params)
        return contents[0] if contents else ""

    def classify_artifact(self, artifact, taxonomy, method):
        """LLM taxonomy classification for fixed/dynamic map-elites."""
        taxonomy_text = "\n".join(
            f"- {name}: {description}"
            for name, description in (taxonomy or {}).items()
        ) or "(no classes yet)"
        mode_instruction = (
            "Choose one of the existing classes."
            if method == "fixed"
            else "Choose an existing class, or propose a new class if none fits."
        )
        prompt_data = {
            "artifact": artifact,
            "new_artifact": artifact,
            "taxonomy_text": taxonomy_text,
            "method": method,
            "mode_instruction": mode_instruction,
        }
        if self.task_description:
            prompt_data["task_description"] = self.task_description

        for _ in range(max(1, self.llm_retries)):
            try:
                messages = create_prompt("core", prompt_data, self.classify_config)
                raw = self.call_llm(messages)
                payload = extract_response(raw, "json")
                data = json.loads(payload)
                if isinstance(data, dict):
                    return data
            except (LLMError, json.JSONDecodeError, TypeError, ValueError):
                continue
            except Exception as exc:  # noqa: BLE001 - classifier must not kill the run
                if self.logger:
                    self.logger.warning("taxonomy classifier failed: %s", exc)
                continue
        return {"class": "unclassified", "new": False, "description": ""}

    def embed_artifact(self, artifact):
        if not self.embedding_model:
            raise ValueError("embedding_model is required for cluster-metric=embedding")
        vectors = self.server.embed(
            [artifact],
            self.embedding_model,
            url=self.embedding_url,
            api_key=self.embedding_api_key,
        )
        return vectors[0]

    # ------------------------------------------------------------------
    # evaluation + artefacts
    # ------------------------------------------------------------------
    def eval_and_save(
        self,
        iteration_idx,
        response_data,
        prompt_data,
        additional_index,
        is_debug=False,
    ):
        solutions_dir = self.get_solutions_dir(
            iteration_idx, additional_index
        )
        solutions_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_debug" if is_debug else ""
        target_path = solutions_dir / (
            f"core_response{suffix}.{self.artifact_extension}"
        )
        write_file(solutions_dir, target_path.name, response_data.get("core_response", ""))
        self.save_response_data(solutions_dir, response_data, is_debug=is_debug)

        stdout_path = solutions_dir / f"eval_stdout{suffix}.txt"
        stderr_path = solutions_dir / f"eval_stderr{suffix}.txt"
        eval_result = run_evaluation(
            self.evaluate_path,
            target_path,
            reference_path=self.reference_path,
            timeout=self.eval_timeout,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

        prompt_data["scores"] = eval_result["scores"]
        prompt_data["result"] = summarize_result(eval_result)
        prompt_data["score"] = score_of(
            eval_result["scores"], key=self.score_key
        )
        prompt_data["rank_score"] = rank_score_of(
            eval_result["scores"], key=self.score_key, minimize=self.minimize
        )
        for key in _FEEDBACK_KEYS:
            prompt_data.pop(key, None)
        if has_error_signal(eval_result):
            prompt_data["error"] = (
                eval_detail(eval_result) or prompt_data["result"]
            )
        else:
            prompt_data.pop("error", None)

        self.feedback_strategy.apply(
            prompt_data,
            eval_result,
            response_data.get("core_response", ""),
            score_key=self.score_key,
        )

        write_file(
            solutions_dir,
            f"scores{suffix}.json",
            json.dumps(eval_result["scores"], indent=2),
        )
        self._event(
            "eval",
            iteration=iteration_idx,
            sample=additional_index,
            debug=is_debug,
            status=eval_result["status"],
            returncode=eval_result.get("returncode"),
            score=prompt_data.get("score"),
            rank_score=prompt_data.get("rank_score"),
            failure_stage=prompt_data.get("failure_stage"),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

        return solutions_dir, prompt_data, target_path, eval_result

    def save_response_data(self, solutions_dir, response_data, is_debug=False):
        suffix = "_debug" if is_debug else ""
        for key, value in response_data.items():
            if key == "core_response":
                continue  # written by eval_and_save under the right name
            if key.endswith("_prompt") or isinstance(value, (dict, list)):
                ending = ".json"
                content = json.dumps(value, indent=2)
            else:
                ending = ".txt"
                content = value if isinstance(value, str) else str(value)
            write_file(solutions_dir, f"{key}{suffix}{ending}", content)

    def log_candidates(
        self, iteration_idx, population, additional_index, **kwargs
    ):
        solutions_dir = self.get_solutions_dir(iteration_idx, additional_index)
        write_file(solutions_dir, "candidates.txt", str(population))

    def get_solutions_dir(self, iteration_idx, additional_index=0):
        appendix = f"iteration_it_{iteration_idx}_samp_{additional_index}"
        return Path(self.experiments_dir) / "solutions" / appendix

    def get_prompt_data(self):
        prompt_data = {}
        if self.task_description:
            prompt_data["task_description"] = self.task_description
        if self.reference_path:
            reference = read_file(self.reference_path)
            # Both keys are set so old and new prompt configs work unchanged.
            prompt_data["reference"] = reference
            prompt_data["reference_result"] = reference
            prompt_data["reference_artifact"] = reference
        return prompt_data

    def _save_summary(self, completed=True):
        best = None
        if self.history:
            best = max(
                self.history,
                key=lambda item: item.get("rank_score", item.get("score", float("-inf"))),
            )
        self.summary = {
            "best": best,
            "history": self.history,
            "completed": bool(completed),
            "map_elites_taxonomy": getattr(self.population, "taxonomy", None),
        }
        write_file(
            self.experiments_dir, "summary.json", json.dumps(self.summary, indent=2)
        )
        if best:
            print(f"\nBest score: {best['score']} -> {best['path']}")

    def __repr__(self):
        return (
            f" num_iterations={self.num_iterations}\n"
            f" num_debug_iterations={self.num_debug_iterations}\n"
            f" num_samples={self.num_samples}\n"
            f" predecessor_selection_method={self.predecessor_selection_method}\n"
            f" max_candidates={self.max_candidates}\n"
            f" minimize={self.minimize}\n"
            f" num_map_elites={self.num_map_elites}\n"
            f" map_elites_method={self.map_elites_method}\n"
            f" map_elites_cluster_metric={self.map_elites_cluster_metric}\n"
            f" diff_based={self.diff_based}\n"
            f" diff_mode={self.diff_mode}\n"
            f" use_strategy={self.use_strategy}\n"
            f" feedback_strategy={self.feedback_strategy.name}\n"
            f" feedback_llm_mode={self.feedback_strategy.llm_mode}\n"
            f" use_feedback={self.use_feedback}\n"
            f" use_debug={self.use_debug}\n"
            f" model_name={self.model_name}\n"
        )


if __name__ == "__main__":
    Agent().run(
        baseline="examples/test_example/baseline.py",
        evaluate="examples/test_example/evaluate.py",
        reference="examples/test_example/reference.py",
    )
