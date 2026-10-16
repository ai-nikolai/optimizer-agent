import json
import argparse

from optimizer_agent.agent import Agent
from optimizer_agent.prompt_config import PromptConfig

# ANSI escape codes for colors and bold
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizer Agent CLI")

    # General args
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--num-debug-iterations", type=int, default=3, help="Number of debug iterations")
    parser.add_argument("--num-islands", type=int, default=1, help="Number of islands")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of beams")

    # Prompts related
    parser.add_argument("--prompt-config-file", type=str, help="Path to prompt configuration file, otherwise default is used.")
    parser.add_argument("--use-kb-prompt", action="store_true", help="Use the KernelBench (TOML-driven) prompt style instead of the simple JSON style.")
    parser.add_argument("--kb-backend", type=str, default="cuda", help="KernelBench backend: cuda, triton, cute, tilelang.")
    parser.add_argument("--kb-option", type=str, default="one_shot", help="KernelBench example option: zero_shot, one_shot, few_shot.")
    parser.add_argument("--kb-gpu", type=str, default="H200", help="KernelBench GPU name (from hardware/gpu_specs.py).")
    parser.add_argument("--kb-precision", type=str, default="fp32", help="KernelBench precision: fp32, fp16, bf16.")

    # Map elites related
    parser.add_argument("--predecessor-selection-method", type=str, default="bestN_last", help="Selection Algorithm", choices=["bestN_last","bestN_all","map_elites"])
    parser.add_argument("--map-elites-method", type=str, default=None, help="Map elites method: cluster, fixed, dynamic ideas", choices=["cluster", "fixed", "dynamic"])
    parser.add_argument("--num-map-elites", type=int, default=1, help="Number of map elites")

    # Sub-agents
    parser.add_argument("--use-strategy", action="store_true", help="Use strategy sub-agent")
    parser.add_argument("--use-feedback", action="store_true", help="Use feedback sub-agent")
    parser.add_argument("--use-debug", action="store_true", help="Use debug sub-agent")
    parser.add_argument("--diff-based", action="store_true", help="Use diffs only... (need to think how to implement)")

    # Feedback strategy (result-driven). KB uses "error_case"; the simple style
    # uses "generic"/"none"/"score".
    parser.add_argument("--feedback-strategy", type=str, default="generic", help="Result-driven feedback strategy: none, generic, error_case, score.")

    # Model related
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000", help="Model URL")
    parser.add_argument("--model-name", type=str, default="default", help="Model name to use at the server")
    parser.add_argument("--model-api-key", type=str, default=None, help="Model API key")
    parser.add_argument("--model-params", type=json.loads, default={}, help="Model parameters as JSON string, e.g. '{\"temperature\": 0.1}'")

    # testing the agent
    parser.add_argument("--test", action="store_true", help="Test the engine")

    # Running the agent
    parser.add_argument("--baseline", type=str, required=True, help="Baseline Function")
    parser.add_argument("--evaluate", type=str, required=True, help="Evaluate function")
    parser.add_argument("--reference", type=str, default="", help="Reference function (optional), gets passed to the evaluate function if set.")

    parser.add_argument("--experiments-dir", type=str, default="opt_agent_experiment", help="The experiments dir for the session.")

    args = parser.parse_args()
    return args


def _default_prompt_path() -> str:
    """Return the bundled simple-style default prompt config path."""
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "test_example", "prompt_config.json")


def main():
    args = parse_args()

    # Load the prompt config (simple or kernel_bench).
    config_path = args.prompt_config_file or _default_prompt_path()
    use_kb = args.use_kb_prompt

    kb_kwargs = {
        "backend": args.kb_backend,
        "option": args.kb_option,
        "gpu_name": args.kb_gpu,
        "precision": args.kb_precision,
    }
    prompt_config = PromptConfig.from_file(config_path, use_kb_prompt=use_kb, **kb_kwargs)
    agent_kwargs = prompt_config.to_prompt_data()

    agent = Agent(
        num_iterations=args.num_iterations,
        num_debug_iterations=args.num_debug_iterations,
        num_islands=args.num_islands,
        num_samples=args.num_samples,
        predecessor_selection_method=args.predecessor_selection_method,
        num_map_elites=args.num_map_elites,
        map_elites_method=args.map_elites_method,
        use_strategy=args.use_strategy,
        use_feedback=args.use_feedback,
        use_debug=args.use_debug,
        feedback_strategy_name=args.feedback_strategy,
        model_name=args.model_name,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_params=args.model_params,
        test=args.test,
        **agent_kwargs,
    )

    agent.run(
        baseline=args.baseline,
        evaluate=args.evaluate,
        reference=args.reference,
        experiments_dir=args.experiments_dir,
    )


if __name__ == "__main__":
    main()
