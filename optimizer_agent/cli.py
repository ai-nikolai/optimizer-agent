"""Command-line entry point for the optimizer agent.

Usage (generic -- nothing CUDA-specific here):

    optimizer-agent --baseline b.py --evaluate e.py [--reference r.py] ...

The agent only needs three things from you: the baseline code to improve, the
evaluator script (which scores candidates), and optionally a reference.
Everything else is search configuration.
"""

import argparse
import json
from pathlib import Path

from optimizer_agent.agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizer Agent CLI")

    # General search configuration
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--num-debug-iterations", type=int, default=0, help="Debug regenerations per failed candidate")
    parser.add_argument("--use-debug", action="store_true", help="Run the debug loop for failed candidates (sets num-debug-iterations=1)")
    parser.add_argument("--num-islands", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per predecessor")

    # Prompts
    parser.add_argument("--prompt-config-file", type=str, default=None,
                        help="Path to a JSON prompt config (system/user shape).")

    # Selection
    parser.add_argument("--predecessor-selection-method", type=str, default="bestN_last",
                        choices=["bestN_last", "bestN_all", "map_elites"],
                        help="Selection algorithm")
    parser.add_argument("--num-map-elites", type=int, default=1, help="Number of map elites")

    # Feedback
    parser.add_argument("--feedback-strategy", type=str, default="v2",
                        help="Result-driven feedback strategy: none, v2")

    # Model
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000", help="Model URL")
    parser.add_argument("--model-name", type=str, default="default", help="Model name to use at the server")
    parser.add_argument("--model-api-key", type=str, default=None, help="Model API key")
    parser.add_argument("--model-params", type=json.loads, default={},
                        help="Model parameters as JSON, e.g. '{\"temperature\": 0.1}'")

    # Testing the agent
    parser.add_argument("--test", action="store_true", help="Test the engine (uses a fake LLM, no network)")

    # Running the agent
    parser.add_argument("--baseline", type=str, required=True, help="Baseline implementation")
    parser.add_argument("--evaluate", type=str, required=True, help="Evaluator script")
    parser.add_argument("--reference", type=str, default="", help="Reference implementation (optional)")
    parser.add_argument("--experiments-dir", type=str, default="opt_agent_experiment",
                        help="The experiments dir for the session.")

    return parser.parse_args()


def _default_prompt_config_file() -> str:
    return str(Path(__file__).resolve().parents[1] / "examples" / "test_example" / "prompt_config.json")


def main():
    args = parse_args()

    # --use-debug is the friendly boolean form of num-debug-iterations.
    num_debug_iterations = args.num_debug_iterations
    if args.use_debug:
        num_debug_iterations = 1

    prompt_config = {}
    cfg_path = args.prompt_config_file or _default_prompt_config_file()
    if Path(cfg_path).exists():
        prompt_config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    agent = Agent(
        num_iterations=args.num_iterations,
        num_debug_iterations=num_debug_iterations,
        num_islands=args.num_islands,
        num_samples=args.num_samples,
        prompt_config=prompt_config,
        predecessor_selection_method=args.predecessor_selection_method,
        num_map_elites=args.num_map_elites,
        feedback_strategy=args.feedback_strategy,
        model_name=args.model_name,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_params=args.model_params,
        test=args.test,
    )

    agent.run(
        baseline=args.baseline,
        evaluate=args.evaluate,
        reference=args.reference,
        experiments_dir=args.experiments_dir,
    )


if __name__ == "__main__":
    main()
