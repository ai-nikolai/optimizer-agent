"""Command-line entry point for the optimizer agent.

The required inputs are the baseline artifact and the evaluator; everything else
is search configuration:

    optimizer-agent --baseline b.py --evaluate e.py [--reference r.py] ...
"""

import argparse
import json

from optimizer_agent.main import Agent


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizer Agent CLI")

    # General search configuration
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of iterations (default: task config or 1)",
    )
    parser.add_argument(
        "--num-debug-iterations",
        type=int,
        default=None,
        help="Debug regenerations per failed candidate (default: task config or 0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples per predecessor (default: task config or 1)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Bounded candidate pool size (default: task config or 20)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=None,
        help="Treat the selection score as lower-is-better.",
    )
    parser.add_argument(
        "--diff-based",
        action="store_true",
        default=None,
        help="Ask the model for edits/diffs against the current artifact.",
    )
    parser.add_argument(
        "--diff-mode",
        type=str,
        default=None,
        choices=["unified-diff", "json"],
        help="Diff format (default: unified-diff).",
    )

    # Prompts
    parser.add_argument(
        "--prompt-config-file",
        type=str,
        default=None,
        help="Path to a JSON prompt config; built-in defaults are used if omitted.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Optional task description exposed to prompts as {task_description}.",
    )

    # Generic feedback / artifact controls
    parser.add_argument(
        "--feedback-strategy",
        type=str,
        default=None,
        choices=["none", "prompt", "rules", "rules_llm", "llm", "evaluator"],
        help="Feedback strategy; overrides prompt/feedback config.",
    )
    parser.add_argument(
        "--feedback-config-file",
        type=str,
        default=None,
        help="JSON file with strategy/profile/hints/metric for a specific task.",
    )
    parser.add_argument(
        "--feedback-profile",
        type=str,
        default=None,
        choices=["generic", "kernel"],
        help="Feedback wording: generic (default, any task) or kernel.",
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default=None,
        choices=["last_code", "last", "largest", "code", "python", "raw", "text", "json"],
        help="How to extract the artifact from a model response (default: last_code).",
    )
    parser.add_argument(
        "--artifact-extension",
        type=str,
        default=None,
        help="File extension for the candidate artifact (default: py).",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default=None,
        help="Evaluator JSON key used as the selection score (default: combined_score).",
    )

    # Selection
    parser.add_argument(
        "--predecessor-selection-method",
        type=str,
        default=None,
        choices=["bestN_last", "bestN_all", "map_elites"],
        help="Selection algorithm (default: task config or bestN_last)",
    )
    parser.add_argument(
        "--num-map-elites",
        type=int,
        default=None,
        help="Number of map elites (default: task config or 1)",
    )
    parser.add_argument(
        "--map-elites-method",
        type=str,
        default=None,
        choices=["cluster", "fixed", "dynamic"],
        help="Map-elites method (default: cluster).",
    )
    parser.add_argument(
        "--map-elites-cluster-metric",
        type=str,
        default=None,
        choices=["jaccard", "embedding"],
        help="Cluster similarity metric (default: jaccard).",
    )
    parser.add_argument(
        "--map-elites-taxonomy-file",
        type=str,
        default=None,
        help="JSON map of class name -> description for fixed/dynamic map-elites.",
    )
    parser.add_argument(
        "--map-elites-cluster-threshold",
        type=float,
        default=None,
        help="Similarity threshold for creating a new cluster (default: 0.2).",
    )
    parser.add_argument(
        "--max-map-elites-cells",
        type=int,
        default=None,
        help="Maximum dynamic taxonomy cells (default: 100).",
    )
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default=None)

    # Sub-agents
    parser.add_argument(
        "--use-strategy", action="store_true", default=None, help="Use the strategy stage"
    )
    parser.add_argument(
        "--use-feedback",
        action="store_true",
        default=None,
        help="Add an LLM reviewer on top of the selected feedback strategy (legacy alias).",
    )
    parser.add_argument(
        "--use-debug",
        action="store_true",
        default=None,
        help="Regenerate failed candidates",
    )
    parser.add_argument(
        "--use-debug-strategy",
        action="store_true",
        default=None,
        help="Use strategy during debug",
    )
    parser.add_argument(
        "--use-debug-feedback",
        action="store_true",
        default=None,
        help="Use feedback during debug",
    )
    # Model
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000", help="Model URL")
    parser.add_argument("--model-name", type=str, default="default", help="Model name at the server")
    parser.add_argument("--model-api-key", type=str, default=None, help="Model API key")
    parser.add_argument(
        "--model-params",
        type=json.loads,
        default={},
        help='Extra model parameters as JSON, e.g. \'{"temperature": 0.1}\'',
    )

    # Evaluation
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=None,
        help="Evaluator subprocess timeout in seconds (default: task config or 600)",
    )

    # LLM reliability / run lifecycle
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-retries", type=int, default=None)
    parser.add_argument("--llm-retry-backoff", type=float, default=None)
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help="Resume a completed experiment (default: on).",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Do not resume; rerun the experiment.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=None,
        help="Delete the experiment directory before starting.",
    )

    # Testing
    parser.add_argument("--test", action="store_true", help="Use a fake LLM (no network)")

    # Running the agent
    parser.add_argument("--baseline", type=str, required=True, help="Baseline artifact/implementation")
    parser.add_argument("--evaluate", type=str, required=True, help="Evaluator script")
    parser.add_argument(
        "--reference", type=str, default="", help="Reference artifact/implementation (optional)"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="opt_agent_experiment",
        help="Directory for logs and candidates",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    agent = Agent(
        num_iterations=args.num_iterations,
        num_debug_iterations=args.num_debug_iterations,
        num_samples=args.num_samples,
        max_candidates=args.max_candidates,
        minimize=args.minimize,
        diff_based=args.diff_based,
        diff_mode=args.diff_mode,
        prompt_config_file=args.prompt_config_file,
        predecessor_selection_method=args.predecessor_selection_method,
        num_map_elites=args.num_map_elites,
        map_elites_method=args.map_elites_method,
        map_elites_cluster_metric=args.map_elites_cluster_metric,
        map_elites_taxonomy_file=args.map_elites_taxonomy_file,
        map_elites_cluster_threshold=args.map_elites_cluster_threshold,
        max_map_elites_cells=args.max_map_elites_cells,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        llm_timeout=args.llm_timeout,
        llm_retries=args.llm_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        resume=args.resume,
        overwrite=args.overwrite,
        use_strategy=args.use_strategy,
        use_feedback=args.use_feedback,
        use_debug=args.use_debug,
        use_debug_strategy=args.use_debug_strategy,
        use_debug_feedback=args.use_debug_feedback,
        model_name=args.model_name,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_params=args.model_params,
        eval_timeout=args.eval_timeout,
        test=args.test,
        feedback_strategy=args.feedback_strategy,
        feedback_config_file=args.feedback_config_file,
        feedback_profile=args.feedback_profile,
        response_format=args.response_format,
        artifact_extension=args.artifact_extension,
        score_key=args.score_key,
        task_description=args.task_description,
    )

    agent.run(
        baseline=args.baseline,
        evaluate=args.evaluate,
        reference=args.reference,
        experiments_dir=args.experiments_dir,
    )


if __name__ == "__main__":
    main()
