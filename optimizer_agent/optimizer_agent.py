import os
import json
import time

import uuid
from pathlib import Path

from . import utils
from . import prompts

class Agent():

    def __init__(self,
        # general args
        num_iterations=1, #total "generations"
        num_debug_iterations=0, #num debug steps between generations
        num_islands=1, #completely independent runs (maybe with cross-over later)
        num_beams=1, #num generations from one predecessor/predecessor group
        # prompts related:
        prompt_config_file=None,
        # predecessor selection related (e.g. elites)
        # num_elites=args.num_elites, #is this needed?
        predecessor_selection_method="bestN_last", #bestN_last (from last generations/iterations)=sequential,parallel, bestN_all (from all generations/iterations)=graph, map_elites, (this operates within a single island)
        num_map_elites=None, #by default topK map elites will be chosen (as dictated by best in elite) # default = None, (as only applicable to map elites) 
        map_elites_method=None, #cluster, fixed, dynamic
        # Sub-agents
        use_debug=False, 
        use_strategy=False, #if map_elites = stratgegy, but this is false, map elites will attempt to classify the strategy by itself.
        use_feedback=False,
        # model related:
        model_name="default",
        model_url="http://0.0.0.0:30000",
        model_api_key="<no_key>",
        model_params=None #dictionary e.g. {'temperature':0.1}
    ):
        """Init func"""
        self.num_iterations=num_iterations
        self.num_debug_iterations=num_debug_iterations
        self.num_islands=num_islands
        self.num_beams=num_beams
        # prompts related:
        self.prompt_config_file=prompt_config_file
        
        self.predecessor_selection_method=predecessor_selection_method
        self.num_map_elites=num_map_elites
        self.map_elites_method=map_elites_method
        # Sub-agents
        self.use_debug=use_debug
        self.use_strategy=use_strategy
        self.use_feedback=use_feedback
        # model related:
        self.model_name=model_name
        self.model_url=model_url
        self.model_api_key=model_api_key
        self.model_params=model_params
    
    def run(self,
        baseline, 
        evaluate,
        reference=None, #this is optional
        experiments_dir="opt_agent_experiment",
    ):
        """The main entry point"""
        self.experiments_dir = Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    

if __name__=="__main__":
    agent=Agent()
    agent.run()
