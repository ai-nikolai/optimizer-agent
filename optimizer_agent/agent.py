"""The optimizer agent: an LLM-driven search over code.

``Agent`` orchestrates the whole thing. It is deliberately agnostic to the
problem being optimised -- it knows nothing about CUDA, kernels, or prompt
*content*. It only manages control flow:

1. pick the best predecessors from the previous generation
   (via :mod:`optimizer_agent.selection`),
2. build prompts and call the model to get new code
   (via :class:`~optimizer_agent.generation.GenerationPipeline`),
3. write the code to disk and score it with the user's evaluator as a
   subprocess (via :mod:`optimizer_agent.evaluation`),
4. enrich ``prompt_data`` with a rule-based feedback string
   (via :mod:`optimizer_agent.prompts`),
5. record everything under ``experiments_dir/``.

Everything domain-specific -- the prompt wording, what a "good" score means,
how correctness is judged -- lives in the user's ``evaluate.py`` and prompt
config. This class is just the loop.

Inspired by the classic LLM optimizer loop (propose -> evaluate -> keep the
best -> repeat) and by projects like open-evolve, which frame it as
generation + reward + elite selection on a diverse population.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from optimizer_agent.evaluation import evaluate_script
from optimizer_agent.generation import GenerationPipeline
from optimizer_agent.llm import FakeLLMServer, LLMServer
from optimizer_agent.prompts import (
    FEEDBACK_STRATEGIES,
    PromptBuilder,
    get_feedback_strategy,
)
from optimizer_agent.selection import _SELECTION_CLASSES, SingleCandidate

# File endings for the artefacts we log per generation.
_FILE_ENDINGS = {
    "core_response": ".py",
    "core_response_raw": ".txt",
    "core_prompt": ".json",
    "scores": ".json",
}


class Agent:
    def __init__(
        self,
        *,
        num_iterations: int = 1,
        num_debug_iterations: int = 0,
        num_islands: int = 1,
        num_samples: int = 1,
        # generic, neutral prompt config (swap wording or pass a file)
        prompt_config: Optional[Dict[str, Any]] = None,
        # predecessor selection
        predecessor_selection_method: str = "bestN_last",
        num_map_elites: int = 1,
        # feedback
        feedback_strategy: str = "v2",
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
        self.prompt_config = prompt_config or {}

        if predecessor_selection_method not in _SELECTION_CLASSES:
            raise NotImplementedError(
                f"Predecessor Selection {predecessor_selection_method!r} not allowed."
            )
        self.predecessor_selection_method = predecessor_selection_method
        self._population_class = _SELECTION_CLASSES[predecessor_selection_method]
        self.num_map_elites = num_map_elites

        if feedback_strategy not in FEEDBACK_STRATEGIES:
            raise ValueError(f"Unknown feedback strategy {feedback_strategy!r}")
        self.feedback_strategy = get_feedback_strategy(feedback_strategy)

        self.model_name = model_name
        self.model_url = model_url
        self.model_api_key = model_api_key
        self.model_params = model_params or {}
        self.server = FakeLLMServer(model_url, model_api_key) if test else LLMServer(model_url, model_api_key)

        self.experiments_dir: Optional[Path] = None
        self.baseline_path: Optional[str] = None
        self.evaluate_path: Optional[str] = None
        self.reference_path: Optional[str] = None
        self.candidates: Dict[int, Any] = {}
        self._core_builder = PromptBuilder(self.prompt_config)

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
            population = self._population_class(num_map_elites=self.num_map_elites)
            self.candidates[island_idx] = population

            num_samples = (
                self.num_map_elites
                if self.predecessor_selection_method == "map_elites"
                else self.num_samples
            )
            for idx in range(num_samples):
                prompt_data = self.get_prompt_data()
                prompt_data["current_implementation"] = self._read_file(self.baseline_path)
                candidate = self.run_single_generation(
                    iteration_idx, island_idx, prompt_data, additional_index=idx
                )
                population.update_candidates(candidate, current_iteration=iteration_idx)
                self.log_candidates(
                    iteration_idx, island_idx, population, additional_index=idx
                )
        else:
            population = self.candidates[island_idx]
            print(population)
            num_samples = (
                1
                if self.predecessor_selection_method == "map_elites"
                else self.num_samples
            )
            for idx, (key, candidate) in enumerate(population.get_best().items()):
                for jdx in range(num_samples):
                    actual_index = idx * num_samples + jdx
                    prompt_data = candidate.prompt_data
                    candidate = self.run_single_generation(
                        iteration_idx,
                        island_idx,
                        prompt_data,
                        other=key,
                        additional_index=actual_index,
                    )
                    population.update_candidates(
                        candidate, current_iteration=iteration_idx
                    )
                    self.log_candidates(
                        iteration_idx, island_idx, population, additional_index=actual_index
                    )

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
        pipeline = GenerationPipeline(
            self.server, self._core_builder, model_params=self.model_params
        )
        response_data, prompt_data = pipeline.generate(prompt_data)
        target_file, eval_result = self._eval_and_save(
            iteration_idx, island_idx, response_data, prompt_data, additional_index
        )
        self._apply_feedback(prompt_data, eval_result)

        # Debug loop: regenerate until it runs (or we run out of attempts).
        if eval_result.status != "ok":
            self._debug_loop(pipeline, iteration_idx, island_idx, response_data, prompt_data, additional_index)
            self._apply_feedback(prompt_data, eval_result)

        # Map-elites cell key (harmless for other selection strategies).
        if self.predecessor_selection_method == "map_elites":
            new_generation = response_data.get("core_response") or ""
            if not other:
                other = new_generation
            other, _ = self.candidates[island_idx].get_key(new_generation, other)

        score = eval_result.result.get("combined_score", float("-inf"))
        return SingleCandidate(target_file, {"combined_score": score}, other=other, prompt_data=prompt_data)

    def _debug_loop(self, pipeline, iteration_idx, island_idx, response_data, prompt_data, additional_index):
        """Regenerate + re-evaluate until the candidate runs or attempts run out."""
        for _ in range(max(1, self.num_debug_iterations)):
            dbg_data, prompt_data = pipeline.generate(prompt_data)
            response_data.update(dbg_data)
            _, eval_result = self._eval_and_save(
                iteration_idx, island_idx, response_data, prompt_data, additional_index
            )
            self._apply_feedback(prompt_data, eval_result)
            if eval_result.status == "ok":
                break

    # ------------------------------------------------------------------
    # evaluation + feedback
    # ------------------------------------------------------------------
    def _eval_and_save(self, iteration_idx, island_idx, response_data, prompt_data, additional_index):
        solutions_dir = self._get_solutions_dir(
            iteration_idx, island_idx, additional_index=additional_index
        )
        solutions_dir.mkdir(parents=True, exist_ok=True)

        code = response_data.get("core_response") or response_data.get("core_response_raw") or ""
        solution_file = solutions_dir / "core_response.py"
        self._write_file(solutions_dir, "core_response.py", code)
        self._log_response_data(solutions_dir, response_data)

        eval_result = evaluate_script(
            self.evaluate_path, str(solution_file), reference_path=self.reference_path
        )
        self._write_file(solutions_dir, "scores.json", json.dumps(eval_result.result))
        self._write_file(solutions_dir, "eval_stdout.txt", eval_result.stdout or "")
        self._write_file(solutions_dir, "eval_stderr.txt", eval_result.stderr or "")
        return solution_file, eval_result

    def _apply_feedback(self, prompt_data, eval_result):
        """Let the feedback strategy merge result-driven hints into ``prompt_data``."""
        scores = eval_result.result if eval_result is not None else {}
        prompt_data.update(self.feedback_strategy.feed_back(prompt_data, scores))

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def log_candidates(self, iteration_idx, island_idx, population, additional_index, **kwargs):
        solutions_dir = self._get_solutions_dir(
            iteration_idx, island_idx, additional_index=additional_index
        )
        self._write_file(solutions_dir, "candidates.txt", str(population))

    def _log_response_data(self, solutions_dir: Path, response_data: Dict[str, Any]):
        for key, value in response_data.items():
            ending = _FILE_ENDINGS.get(key, ".txt")
            if key in ("core_prompt", "scores"):
                value = json.dumps(value)
            self._write_file(solutions_dir, f"{key}{ending}", value)

    def _get_solutions_dir(self, iteration_idx, island_idx, additional_index=0):
        appendix = f"iteration_it_{iteration_idx}_is_{island_idx}_samp_{additional_index}"
        return Path(self.experiments_dir) / "solutions" / appendix

    @staticmethod
    def _read_file(filepath):
        with open(filepath, "r", encoding="utf-8") as handle:
            return handle.read()

    def _write_file(self, solutions_dir, file_name, content):
        try:
            Path(solutions_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"failed making folder: {solutions_dir}\nError: {exc}")
        try:
            with open(Path(solutions_dir) / file_name, "w", encoding="utf-8") as handle:
                handle.write(content)
        except Exception as exc:
            print(f"Writing File {solutions_dir}/{file_name} failed")
            print(exc)

    def __repr__(self):
        return (
            f" num_debug_iterations={self.num_debug_iterations}\n"
            f" num_islands={self.num_islands}\n"
            f" num_samples={self.num_samples}\n"
            f" predecessor_selection_method={self.predecessor_selection_method}\n"
            f" num_map_elites={self.num_map_elites}\n"
            f" feedback_strategy={self.feedback_strategy.__class__.__name__}\n"
        )


if __name__ == "__main__":
    agent = Agent(test=True)
    agent.run(baseline=None, evaluate=None)
