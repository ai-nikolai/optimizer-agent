import os
import json
import signal

import time
import uuid

import argparse
import sys

from .optimizer_agent import Agent

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

    # Overall strategy
    parser.add_argument("--overall-strategy", type=str, default="sequential", help="Overall strategy: sequential, parallel, tree, graph...")

    # General args
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--num-debug-iterations", type=int, default=3, help="Number of debug iterations")
    parser.add_argument("--num-islands", type=int, default=1, help="Number of islands")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams")

    # Prompts related
    parser.add_argument("--prompt-config-file", type=str, help="Path to prompt configuration file")

    # Map elites related
    parser.add_argument("--map-elites-method", type=str, default="cluster", help="Map elites method: cluster, fixed, dynamic ideas", choices=["cluster", "fixed", "dynamic"])
    parser.add_argument("--num-elites", type=int, default=3, help="Number of elites")
        # Sub-agents
    parser.add_argument("--use-debug", action="store_true", help="Use debug sub-agent")
    parser.add_argument("--use-strategy", action="store_true", help="Use strategy sub-agent")
    parser.add_argument("--use-feedback", action="store_true", help="Use feedback sub-agent")
    # Model related
    parser.add_argument("--model-url", type=str, default="http://0.0.0.0:30000", help="Model URL")
    parser.add_argument("--model-name", type=str, default="default", help="Model name to use at the server")
    parser.add_argument("--model-api-key", type=str, default=None, help="Model API key")
    parser.add_argument("--model-params", type=json.loads, default={}, help="Model parameters as JSON string, e.g. '{\"temperature\": 0.1}'")

    # Running the agent
    parser.add_argument("--baseline", type=str, required=True, help="Baseline Function")
    parser.add_argument("--evaluate", type=str, required=True, help="Evaluate function")
    parser.add_argument("--reference", type=str, default="", help="Reference function (optional), gets passed to the evaluate function if set.")

    parser.add_argument("--experiments-dir", type=str, default="opt_agent_experiment" help="The experiments dir for the session.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    agent = Agent(
        # general strategy: sequential, parallel, tree, graph...
        # overall_strategy=args.overall_strategy, #is this needed? or is this implicit in N(iterations), I(islands), B(beams), selection_algorithm
        # general args
        num_iterations=args.num_iterations, #total "generations"
        num_debug_iterations=args.num_debug_iterations, #num debug steps between generations
        num_islands=args.num_islands, #completely independent runs (maybe with cross-over later)
        num_beams=args.num_beams, #num generations from one predecessor/predecessor group
        # prompts related:
        prompt_config_file=args.prompt_config_file,
        # predecessor selection related (e.g. elites)
        # num_elites=args.num_elites, #is this needed?
        predecessor_selection_method=args.predecessor_selection_method, #bestN_last (from last generations/iterations)=sequential,parallel, bestN_all (from all generations/iterations)=graph, map_elites, (this operates within a single island)
        num_map_elites=args.num_map_elites, #by default topK map elites will be chosen (as dictated by best in elite) # default = None, (as only applicable to map elites) 
        map_elites_method=args.map_elites_method, #cluster, fixed, dynamic
        # Sub-agents
        use_debug=args.use_debug, 
        use_strategy=args.use_strategy, #if map_elites = stratgegy, but this is false, map elites will attempt to classify the strategy by itself.
        use_feedback=args.use_feedback,
        # model related:
        model_name=args.model_name,
        model_url=args.model_url,
        model_api_key=args.model_api_key,
        model_params=args.model_params #dictionary e.g. {'temperature':0.1}
    )


    agent.run(
        baseline=args.baseline, 
        evaluate=args.evaluate,
        reference=args.reference, #this is optional
        experiments_dir=args.experiments_dir
    )



if __name__=="__main__":
    main()
