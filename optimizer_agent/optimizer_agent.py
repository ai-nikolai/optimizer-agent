import os
import json
import time

import uuid
from pathlib import Path

from . import utils
from . import prompts
from .candidates import *


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
        diff_based=False,
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
        if predecessor_selection_method=="bestN_all":
            self.predecessor_candidates_class = CandidatesBestNAll
        elif predecessor_selection_method=="bestN_last":
            self.predecessor_candidates_class = CandidatesBestNLast
        elif predecessor_selection_method=="map_elites":
            self.predecessor_candidates_class = MapElitesCandidates
        else:
            raise NotImplementedError(f"Predecessor Selection {predecessor_selection_method} not allowed.")
        
        self.num_map_elites=num_map_elites
        self.map_elites_method=map_elites_method

        # Sub-agents
        self.use_debug=use_debug
        self.use_strategy=use_strategy
        self.use_feedback=use_feedback
        self.diff_based=diff_based

        # model related:
        self.model_name=model_name
        self.model_url=model_url
        self.model_api_key=model_api_key
        self.model_params=model_params

        # RUNTIME
        self.experiments_dir = None
        self.baseline_path = None
        self.evaluate_path = None
        self.reference_path = None

        # Candidates
        self.candidates = {}
    
    def run(self,
        baseline, 
        evaluate,
        reference=None, #this is optional
        experiments_dir="opt_agent_experiment",
    ):
        """The main entry point"""
        self.experiments_dir = Path(experiments_dir).mkdir(parents=True, exist_ok=True)
        
        self.baseline_path = baseline
        
        self.evaluate_path = evaluate
        
        self.reference_path = reference

        for idx in range(self.num_iterations):
            print(f"====================================")
            print(f"\n---- Running Iteration: {idx}/{self.num_iterations} ----")
            print(self)
            self.run_single_iteration(idx)
    

    def run_single_iteration(self, iteration_idx):
        for idx in range(self.num_islands):
            print(f"\n---- Running Island: {idx}/{self.num_islands} ----")
            self.run_single_island(iteration_idx, idx)
    

    def get_prompt_data(self):
        prompt_data = {}
        
        if self.reference_path:
            prompt_data.update({"reference":self._read_file(self.reference_path)})

        return prompt_data


    def run_single_island(self, iteration_idx, island_idx):
        prompt_data = self.get_prompt_data()
        if iteration_idx == 0:
            self.candidates[island_idx] = self.predecessor_candidates_class(num_map_elites=self.num_map_elites)
            prompt_data.update({"current_implementation":self._read_file(self.baseline_path)})
            self.run_single_generation(iteration_idx,island_idx,prompt_data)
        else:
            current_candidates_object = self.candidates[island_idx]

            best_candidates = current_candidates_object.get_best()
            num_candidates = len(best_candidates)

            for key,candidate in best_candidates.items():
                prompt_data.update({"current_implementation":self._read_file(self.candidate.path)})
                self.run_single_generation(iteration_idx,island_idx,prompt_data, key)

    def run_single_generation(self, iteration_idx, island_idx, prompt_data, other=None):
        print(f'---- Running generation {iteration_idx},{island_idx} ----')
        


    def __repr__(self):
        out = ""
        out += f" num_debug_iterations={self.num_debug_iterations}=\n"
        out += f" num_islands={self.num_islands}=\n"
        out += f" num_beams={self.num_beams}=\n"
        # prompts related:
        out += f" prompt_config_file={self.prompt_config_file}=\n"
        
        out += f" predecessor_selection_method={self.predecessor_selection_method}=\n"
        out += f" num_map_elites={self.num_map_elites}=\n"
        out += f" map_elites_method={self.map_elites_method}=\n"
        # Sub-agents
        out += f" use_debug={self.use_debug}=\n"
        out += f" use_strategy={self.use_strategy}=\n"
        out += f" use_feedback={self.use_feedback}=\n"
        out += f" diff_based={self.diff_based}=\n"

        if self.experiments_dir:
            out += f" experiments_dir={self.experiments_dir}=\n"

        return out

    @staticmethod
    def _read_file(filepath):
        """read file as str"""
        with open(filepath, 'r') as file:
            return file.read() 

if __name__=="__main__":
    agent=Agent()
    agent.run()

    print(agent)
