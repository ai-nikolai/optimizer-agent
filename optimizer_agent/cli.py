import json
import argparse
from pathlib import Path

from optimizer_agent.agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizer Agent CLI")

    # General args
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--num-debug-iterations", type=int, default=0, help="Number of debug iterations per failed candidate")
    parser.add_argument("--num-islands", type=int, default=1, help="Number of islands")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per island")

    # Prompt
    parser.add_argument("--prompt-config-file", type=str, default=None,
                        help="Path to a neutral JSON prompt config (system/user shape). "
                             "Defaults to the bundled generic prompt.")

    # Selection
    parser.add_argument("--predecessor-selection-method", type=str, default="bestN_last",
                        help="Selection algorithm",
                        choices=["bestN_last", "bestN_all", "map_elites"])
    parser.add_argument("--num-map-elites", type=int, default=1, help="Number of map elites")

    # Debug + feedback
    parser.add_argument("--use-debug", action="store_true", help="Run the debug re-generation loop on failure")
    parser.add_argument("--feedback-strategy", type=str, default="v2",
                        help="Result-driven feedback strategy: none, v2")

    # Model
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000", help="Model URL")
    parser.add_argument("--model-name", type=str, default="default", help="Model name to use at the server")
    parser.add_argument("--model-api-key", type=str, default=None, help="Model API key")
    parser.add_argument("--model-params", type=json.loads, default={},
                        help="Model parameters as JSON string, e.g. '{\"temperature\": 0.1}'")

    # testing the agent
    parser.add_argument("--test", action="store_true", help="Test the engine (uses a fake LLM, no network)")

    # Running the agent
    parser.add_argument("--baseline", type=str, required=True, help="Baseline implementation")
    parser.add_argument("--evaluate", type=str, required=True, help="Evaluator script")
    parser.add_argument("--reference", type=str, default="", help="Reference implementation (optional)")
    parser.add_argument("--experiments-dir", type=str, default="opt_agent_experiment",
                        help="The experiments dir for the session.")

    return parser.parse_args()


def _default_prompt_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "examples" / "test_example" / "prompt_config.json")


def main():
    args = parse_args()

    # Load the neutral prompt config (if provided); otherwise the agent uses its default.
    prompt_config = {}
    cfg_path = args.prompt_config_file or _default_prompt_path()
    prompt_config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    agent = Agent(
        num_iterations=args.num_iterations,
        num_debug_iterations=args.num_debug_iterations,
        num_islands=args.num_islands,
        num_samples=args.num_samples,
        prompt_config=prompt_config,
        predecessor_selection_method=args.predecessor_selection_method,
        num_map_elites=args.num_map_elites,
        use_debug=args.use_debug,
        feedback_strategy_name=args.feedback_strategy,
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
