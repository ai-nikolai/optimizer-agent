"""The optimizer agent: an LLM-driven search over code.

``Agent`` is the orchestrator. For a fixed number of *iterations* and *islands*
it repeatedly:

1. selects the best candidates from the previous generation (via a pluggable
   selection scheme in :mod:`optimizer_agent.selection`),
2. builds prompts and calls the LLM to produce improved code
   (via :class:`optimizer_agent.generation.GenerationPipeline`),
3. evaluates the candidate by running the user's ``evaluate.py`` as a subprocess
   (via :mod:`optimizer_agent.evaluation`),
4. enriches ``prompt_data`` with a pluggable feedback strategy
   (``--feedback-strategy``),
5. records everything under ``experiments_dir/``.

Two prompt styles are supported through the :class:`PromptBuilder` interface:

* **simple** - a JSON config with ``improve`` / ``debug`` sections, each with
  ``core`` / ``strategy`` / ``feedback`` (this is the original optimizer-agent
  behaviour and the ``prompt_config.json`` shape).
* **kernel_bench** - a TOML-driven prompt (``kb_prompt_config`` with
  ``backend`` / ``option`` / ``gpu_name`` / ``precision``) produced by
  :class:`~optimizer_agent.prompts.kernel_bench.KernelBenchPromptBuilder`.

Both styles share the exact same control flow; only the builders differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from optimizer_agent.generation import GenerationPipeline, ResponseEnum
from optimizer_agent.llm import LLMServer, FakeLLMServer
from optimizer_agent.prompts.base import PromptBuilder
from optimizer_agent.prompts.simple import SimplePromptBuilder
from optimizer_agent.prompts.kernel_bench import KernelBenchPromptBuilder
from optimizer_agent.selection import (
    SingleCandidate,
    CandidatesBestNAll,
    CandidatesBestNLast,
    MapElitesCandidates,
)
from optimizer_agent.evaluation import evaluate_script, EvalResult
from optimizer_agent.feedback import FeedbackStrategy, get_feedback_strategy

# File endings for the artefacts logged for every generation.
_FILE_ENDINGS = {
    "core_response": ".py",
    "core_response_raw": ".txt",
    "core_prompt": ".json",
    "strategy_response_raw": ".txt",
    "strategy_prompt": ".json",
    "feedback_response_raw": ".txt",
    "feedback_prompt": ".json",
    "scores": ".json",
    "eval_stdout": ".txt",
    "eval_stderr": ".txt",
}


class Agent:
    def __init__(
        self,
        *,
        num_iterations: int = 1,
        num_debug_iterations: int = 0,
        num_islands: int = 1,
        num_samples: int = 1,
        # simple-style JSON prompt config (improve / debug sections)
        improve_prompt_config: Optional[Dict[str, Any]] = None,
        debug_prompt_config: Optional[Dict[str, Any]] = None,
        # kernel-bench-style TOML-driven prompt config
        kb_prompt_config: Optional[Dict[str, Any]] = None,
        use_kb_prompt: bool = False,
        # predecessor selection
        predecessor_selection_method: str = "bestN_last",
        num_map_elites: int = 1,
        map_elites_method: Optional[str] = None,
        # feedback / sub-agents
        use_feedback: bool = False,
        use_debug_feedback: bool = False,
        use_strategy: bool = False,
        use_debug: bool = False,
        feedback_strategy: Optional[FeedbackStrategy] = None,
        feedback_strategy_name: Optional[str] = None,
        # model
        model_name: str = "default",
        model_url: str = "http://0.0.0.0:30000",
        model_api_key: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        # smoke-test mode
        test: bool = False,
    ):
        self.num_iterations = num_iterations
        self.num_debug_iterations = num_debug_iterations
        self.num_islands = num_islands
        self.num_samples = num_samples

        self.use_kb_prompt = use_kb_prompt
        self.improve_prompt_config = improve_prompt_config or {}
        self.debug_prompt_config = debug_prompt_config or {}
        self.kb_prompt_config = kb_prompt_config or {}

        self.predecessor_selection_method = predecessor_selection_method
        self._predecessor_candidates_class = self._selection_class(predecessor_selection_method)
        self.num_map_elites = num_map_elites
        self.map_elites_method = map_elites_method

        # Sub-agents / feedback.
        self.use_strategy = use_strategy
        self.use_feedback = use_feedback
        self.use_debug_feedback = use_debug_feedback
        self.use_debug = use_debug
        self.feedback_strategy = (
            feedback_strategy or get_feedback_strategy(feedback_strategy_name or "none")
        )

        # Model.
        self.model_name = model_name
        self.model_url = model_url
        self.model_api_key = model_api_key
        self.model_params = model_params or {}

        self.server = FakeLLMServer(model_url, model_api_key) if test else LLMServer(model_url, model_api_key)

        # Runtime state.
        self.experiments_dir: Optional[Path] = None
        self.baseline_path: Optional[str] = None
        self.evaluate_path: Optional[str] = None
        self.reference_path: Optional[str] = None

        # Builders (populated in _build).
        self._kb_core_builder: Optional[KernelBenchPromptBuilder] = None
        self._core_builder: Optional[PromptBuilder] = None
        self._strategy_builder: Optional[PromptBuilder] = None
        self._feedback_builder: Optional[PromptBuilder] = None
        self._debug_core_builder: Optional[PromptBuilder] = None
        self._debug_strategy_builder: Optional[PromptBuilder] = None
        self._build()

        # Per-island candidate populations.
        self.candidates: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _selection_class(method: str):
        mapping = {
            "bestN_all": CandidatesBestNAll,
            "bestN_last": CandidatesBestNLast,
            "map_elites": MapElitesCandidates,
        }
        if method not in mapping:
            raise NotImplementedError(
                f"Predecessor Selection {method!r} not allowed."
            )
        return mapping[method]

    def _build(self):
        if self.use_kb_prompt:
            self._kb_core_builder = KernelBenchPromptBuilder(self.kb_prompt_config)
            return

        # simple style
        self._core_builder = SimplePromptBuilder(self.improve_prompt_config["improve"]["core"])
        self._strategy_builder = SimplePromptBuilder(self.improve_prompt_config["improve"]["strategy"])
        self._feedback_builder = SimplePromptBuilder(self.improve_prompt_config["improve"]["feedback"])
        self._debug_core_builder = SimplePromptBuilder(self.debug_prompt_config["debug"]["core"])
        self._debug_strategy_builder = SimplePromptBuilder(self.debug_prompt_config["debug"]["strategy"])

    def _make_pipeline(self, *, debug: bool = False) -> GenerationPipeline:
        if self.use_kb_prompt:
            return GenerationPipeline(
                self.server, core_builder=self._kb_core_builder, model_params=self.model_params
            )

        if debug:
            return GenerationPipeline(
                self.server,
                core_builder=self._debug_core_builder,
                strategy_builder=self._debug_strategy_builder,
                model_params=self.model_params,
            )

        return GenerationPipeline(
            self.server,
            core_builder=self._core_builder,
            strategy_builder=self._strategy_builder if self.use_strategy else None,
            feedback_builder=self._feedback_builder if self.use_feedback else None,
            model_params=self.model_params,
        )

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        baseline: str,
        evaluate: str,
        reference: Optional[str] = None,
        experiments_dir: str = "opt_agent_experiment",
    ):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(self.experiments_dir, "metadata.txt", str(self))

        self.baseline_path = baseline
        self.evaluate_path = evaluate
        self.reference_path = reference

        for idx in range(self.num_iterations):
            print("=" * 36)
            print(f"---- Running Iteration: {idx}/{self.num_iterations} ----")
            print(self)
            self.run_single_iteration(idx)

    # ------------------------------------------------------------------
    # iteration / island control flow
    # ------------------------------------------------------------------
    def run_single_iteration(self, iteration_idx: int):
        for idx in range(self.num_islands):
            print(f"\n---- Running Island: {idx}/{self.num_islands} ----")
            self.run_single_island(iteration_idx, idx)

    def get_prompt_data(self) -> Dict[str, Any]:
        prompt_data: Dict[str, Any] = {}
        if self.reference_path:
            prompt_data["reference"] = self._read_file(self.reference_path)
        return prompt_data

    def run_single_island(self, iteration_idx: int, island_idx: int):
        if iteration_idx == 0:
            current = self._predecessor_candidates_class(num_map_elites=self.num_map_elites)
            self.candidates[island_idx] = current

            if self.predecessor_selection_method == "map_elites":
                num_actual_samples = self.num_map_elites
            else:
                num_actual_samples = self.num_samples

            for idx in range(num_actual_samples):
                prompt_data = self.get_prompt_data()
                prompt_data["current_implementation"] = self._read_file(self.baseline_path)
                candidate = self.run_single_generation(
                    iteration_idx, island_idx, prompt_data, additional_index=idx
                )
                current.update_candidates(candidate, current_iteration=iteration_idx)
                self.log_candidates(iteration_idx, island_idx, current, additional_index=idx)
        else:
            current = self.candidates[island_idx]
            print(current)
            best_candidates = current.get_best()

            if self.predecessor_selection_method == "map_elites":
                num_actual_samples = 1
            else:
                num_actual_samples = self.num_samples

            for idx, (key, candidate) in enumerate(best_candidates.items()):
                for jdx in range(num_actual_samples):
                    actual_index = idx * num_actual_samples + jdx
                    prompt_data = candidate.prompt_data
                    candidate = self.run_single_generation(
                        iteration_idx, island_idx, prompt_data,
                        other=key, additional_index=actual_index,
                    )
                    current.update_candidates(candidate, current_iteration=iteration_idx)
                    self.log_candidates(iteration_idx, island_idx, current, additional_index=actual_index)

    # ------------------------------------------------------------------
    # a single generation (with optional debug loop)
    # ------------------------------------------------------------------
    def run_single_generation(
        self,
        iteration_idx: int,
        island_idx: int,
        prompt_data: Dict[str, Any],
        other: Optional[str] = None,
        additional_index: Optional[int] = None,
    ) -> SingleCandidate:
        pipeline = self._make_pipeline(debug=False)
        response_data, prompt_data = pipeline.generate(prompt_data)

        success, eval_result, target_file = self._eval_and_save(
            iteration_idx, island_idx, response_data, prompt_data, additional_index
        )

        scores = eval_result.result if eval_result else {}

        # Result-driven feedback strategy runs for every candidate.
        self._apply_feedback(prompt_data, scores)

        if not success and self.use_debug:
            prompt_data, eval_result, target_file, success = self._debug_loop(
                iteration_idx, island_idx, response_data, prompt_data, additional_index
            )
            scores = eval_result.result if eval_result else {}
            self._apply_feedback(prompt_data, scores)

        # Map-elites key assignment (works for any selection scheme too).
        if self.predecessor_selection_method == "map_elites":
            new_generation = response_data.get("core_response") or ""
            if not other:
                other = new_generation
            map_elites = self.candidates[island_idx]
            other, _ = map_elites.get_key(new_generation, other)

        path = str(target_file) if target_file is not None else ""
        score = scores.get("combined_score", float("-inf"))
        return SingleCandidate(path, {"combined_score": score}, other=other, prompt_data=prompt_data)

    def _debug_loop(self, iteration_idx, island_idx, response_data, prompt_data, additional_index):
        """Re-generate + re-evaluate a fixed number of times until it succeeds."""
        debug_pipeline = self._make_pipeline(debug=True)
        error_text = self._error_text(response_data)
        prompt_data["error"] = error_text

        eval_result: Optional[EvalResult] = None
        target_file = None
        for _ in range(max(1, self.num_debug_iterations)):
            dbg_data, prompt_data = debug_pipeline.generate_debug(prompt_data)
            # merge debug response files into the logged artefacts
            response_data.update(dbg_data)
            success, eval_result, target_file = self._eval_and_save(
                iteration_idx, island_idx, response_data, prompt_data, additional_index
            )
            scores = eval_result.result if eval_result else {}
            self._apply_feedback(prompt_data, scores)
            if success:
                break
        return prompt_data, eval_result, target_file, bool(eval_result and eval_result.compiled and eval_result.correctness)

    # ------------------------------------------------------------------
    # evaluation + feedback
    # ------------------------------------------------------------------
    def _eval_and_save(self, iteration_idx, island_idx, response_data, prompt_data, additional_index):
        solutions_dir = self._get_solutions_dir(iteration_idx, island_idx, additional_index=additional_index)
        solutions_dir.mkdir(parents=True, exist_ok=True)

        # Log the generated artefacts (strategy/core prompts + responses + code).
        code = response_data.get("core_response") or response_data.get("core_response_raw") or ""
        self._write_file(solutions_dir, "core_response.py", code)
        self._log_response_data(solutions_dir, response_data)

        # Run the user evaluator as a subprocess and parse its JSON.
        eval_result = evaluate_script(
            self.evaluate_path,
            solutions_dir / "core_response.py",
            reference_path=self.reference_path,
        )

        self._write_file(
            solutions_dir,
            "scores.json",
            json.dumps(eval_result.result),
        )
        self._write_file(solutions_dir, "eval_stdout.txt", eval_result.stdout or "")
        self._write_file(solutions_dir, "eval_stderr.txt", eval_result.stderr or "")

        success = bool(eval_result.compiled and eval_result.correctness)
        return success, eval_result, solutions_dir / "core_response.py"

    def _apply_feedback(self, prompt_data, scores):
        """Merge result-driven feedback into ``prompt_data``.

        The strategy writes the structured fields (``feedback``, ``error_case``,
        ``score`` ...) plus ``previous_generation``. The LLM feedback sub-agent,
        when enabled, adds ``feedback_prompt`` / ``feedback_response_raw`` via the
        generation pipeline - those are already logged in ``_eval_and_save``.
        """
        # Forward the current model text as the "previous generation" for KB.
        prompt_data["previous_generation"] = prompt_data.get("new_implementation") or prompt_data.get("core_response") or ""
        # Remember the previous score so a score-based strategy can compare.
        if "combined_score" in scores:
            prompt_data["prev_score"] = scores["combined_score"]
        enriched = self.feedback_strategy.feed_back(prompt_data, scores)
        prompt_data.update(enriched)

    @staticmethod
    def _error_text(response_data: Dict[str, Any]) -> str:
        """Best-effort error text for the debug loop."""
        raw = response_data.get("core_response_raw", "")
        if not raw:
            return ""
        # Prefer a fenced ```...``` block if present.
        idx = raw.find("```")
        if idx != -1:
            end = raw.find("```", idx + 3)
            if end != -1:
                return raw[idx + 3:end].strip()
        return raw

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def log_candidates(self, iteration_idx, island_idx, current, additional_index, **kwargs):
        value = str(current)
        solutions_dir = self._get_solutions_dir(iteration_idx, island_idx, additional_index=additional_index)
        self._write_file(solutions_dir, "candidates.txt", value)

    def _log_response_data(self, solutions_dir: Path, response_data: Dict[str, Any]):
        for key, value in response_data.items():
            ending = _FILE_ENDINGS.get(key, ".txt")
            if key in ("core_prompt", "strategy_prompt", "feedback_prompt"):
                value = json.dumps(value)
            elif key == "scores":
                value = json.dumps(value)
            self._write_file(solutions_dir, f"{key}{ending}", value)

    def _get_solutions_dir(self, iteration_idx, island_idx, additional_index=0):
        appendix = f"iteration_it_{iteration_idx}_is_{island_idx}_samp_{additional_index}"
        return Path(self.experiments_dir) / "solutions" / appendix

    @staticmethod
    def _read_file(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, solutions_dir, file_name, content):
        try:
            Path(solutions_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"failed making folder: {solutions_dir}\nError: {e}")
        try:
            with open(Path(solutions_dir) / file_name, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Writing File {solutions_dir}/{file_name} failed")
            print(e)

    def __repr__(self):
        out = ""
        out += f" num_debug_iterations={self.num_debug_iterations}\n"
        out += f" num_islands={self.num_islands}\n"
        out += f" num_samples={self.num_samples}\n"
        out += f" prompt_config_file={self._prompt_config_label()}\n"
        out += f" predecessor_selection_method={self.predecessor_selection_method}\n"
        out += f" num_map_elites={self.num_map_elites}\n"
        out += f" map_elites_method={self.map_elites_method}\n"
        out += f" use_debug={self.use_debug}\n"
        out += f" use_strategy={self.use_strategy}\n"
        out += f" use_feedback={self.use_feedback}\n"
        out += f" feedback_strategy={type(self.feedback_strategy).__name__}\n"
        out += f" use_kb_prompt={self.use_kb_prompt}\n"
        if self.experiments_dir:
            out += f" experiments_dir={self.experiments_dir}\n"
        return out

    def _prompt_config_label(self):
        if self.use_kb_prompt:
            return f"kernel_bench ({self.kb_prompt_config.get('backend', 'cuda')})"
        if isinstance(self.improve_prompt_config, dict) and self.improve_prompt_config.get("path"):
            return self.improve_prompt_config["path"]
        if isinstance(self.improve_prompt_config, dict):
            return "simple (inline)"
        return ""


if __name__ == "__main__":
    from optimizer_agent.prompt_config import PromptConfig

    agent = Agent(test=True)
    agent.run(baseline=None, evaluate=None)
