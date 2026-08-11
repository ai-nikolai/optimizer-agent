import os
import json
import time

import tempfile

import subprocess

import uuid
from pathlib import Path


from optimizer_agent.candidates import (
    SingleCandidate,
    CandidatesBestNAll,
    CandidatesBestNLast,
    MapElitesCandidates
)
from optimizer_agent.baseagents import (
    CoreAgent,
    ResponseEnum
)

from optimizer_agent.utils_llm import (
    LLMServer,
    FakeLLMServer
)

from optimizer_agent.utils import (
    extract_eval_results,
)


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
        use_strategy=False, #if map_elites = stratgegy, but this is false, map elites will attempt to classify the strategy by itself.
        use_feedback=False,
        diff_based=False,
        use_debug=False, 
        use_debug_strategy=False,
        use_debug_feedback=False,
        # model related:
        model_name="default",
        model_url="http://0.0.0.0:30000",
        model_api_key="<no_key>",
        model_params=None, #dictionary e.g. {'temperature':0.1}
        # if we should test the agent (fake server)
        test=False
    ):
        """Init func"""
        self.num_iterations=num_iterations
        self.num_debug_iterations=num_debug_iterations
        self.num_islands=num_islands
        self.num_beams=num_beams
        # prompts related:
        self.prompt_config_file=prompt_config_file

        with open(self.prompt_config_file, 'r', encoding='utf-8') as f:
            self.prompt_config = json.load(f)
        
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
        self.use_strategy=use_strategy
        self.use_feedback=use_feedback

        self.use_debug=use_debug
        self.use_debug_strategy=use_debug_strategy,
        self.use_debug_feedback=use_debug_feedback,

        self.diff_based=diff_based

        # model related:
        self.model_name=model_name
        self.model_url=model_url
        self.model_api_key=model_api_key
        self.model_params=model_params

        if test:
            self.server = FakeLLMServer(model_url, model_api_key)
        else:
            self.server = LLMServer(model_url, model_api_key)

        # RUNTIME
        self.experiments_dir = None
        self.baseline_path = None
        self.evaluate_path = None
        self.reference_path = None

        # Candidates
        self.candidates = {}

        self.core_agent = CoreAgent(self.server, prompt_config=self.prompt_config["improve"], use_strategy=use_strategy, use_feedback=use_feedback)
        
        if use_debug:
            self.debug_agent = CoreAgent(self.server, prompt_config=self.prompt_config["debug"], use_strategy=use_debug_strategy, use_feedback=use_debug_feedback)
        else:
            self.debug_agent = None


    
    def run(self,
        baseline, 
        evaluate,
        reference=None, #this is optional
        experiments_dir="opt_agent_experiment",
    ):
        """The main entry point"""
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = baseline
        self.evaluate_path = evaluate
        self.reference_path = reference

        for idx in range(self.num_iterations):
            print(f"====================================")
            print(f"---- Running Iteration: {idx}/{self.num_iterations} ----")
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
        if iteration_idx == 0:
            prompt_data = self.get_prompt_data()
            current_candidates_object = self.predecessor_candidates_class(num_map_elites=self.num_map_elites)
            self.candidates[island_idx] = current_candidates_object
            prompt_data.update({"current_implementation":self._read_file(self.baseline_path)})
            
            candidate = self.run_single_generation(iteration_idx,island_idx,prompt_data)
            current_candidates_object.update_candidates(candidate, current_iteration=iteration_idx,)
        else:
            current_candidates_object = self.candidates[island_idx]

            print(current_candidates_object)

            best_candidates = current_candidates_object.get_best()
            num_candidates = len(best_candidates)

            for key,candidate in best_candidates.items():
                prompt_data = candidate.prompt_data
                candidate = self.run_single_generation(iteration_idx,island_idx,prompt_data, key)
                # other needs to be cluster info for map_elites...
                current_candidates_object.update_candidates(candidate, current_iteration=iteration_idx,)

    def run_single_generation(self, iteration_idx, island_idx, prompt_data, other=None):
        """ 
        A single generation with debugging...

        TODO: Change this to a strategy as well. Similar to CoreAgent, such that this can be easily changed...
        
        """
        response_data, prompt_data = self.core_agent.generate_response(prompt_data) 
        response_data, prompt_data, target_file_name, success  = self.run_eval_and_safe(iteration_idx, island_idx, response_data, prompt_data)

        if not success and self.use_debug:
            for idx in range(self.num_debug_iterations):
                print(f"Debug: {idx} -- {iteration_idx},{island_idx},{other}")
        
        # Return working candidate
        candidate = SingleCandidate(target_file_name, prompt_data["scores"], other=other, prompt_data=prompt_data)

        return candidate


    def run_eval_and_safe(self, iteration_idx, island_idx, response_data, prompt_data, is_debug=False):
        solutions_dir, log_success = self.log_response_data(response_data, iteration_idx, island_idx, is_debug=is_debug)
        
        file_name = ResponseEnum.CORE_RESPONSE.value+".py"

        target_file_name = solutions_dir / file_name

        scores, stdout, stderr, success = self._run_eval(target_file_name)
        
        response_data[ResponseEnum.SCORES.value] = scores
        response_data[ResponseEnum.EVAL_STDOUT.value] = stdout
        response_data[ResponseEnum.EVAL_STDERR.value] = stderr

        prompt_data["scores"] = scores

        _, log_success = self.log_response_data(response_data, iteration_idx, island_idx, is_debug=is_debug)

        return response_data, prompt_data, target_file_name, success

    def log_response_data(self, response_data, iteration_idx, island_idx, is_debug=False):
        """All sorts of saving to files..."""
        # Determine the file name based on whether it's debug or not
        # print(f"we are here::{self.experiments_dir},,{response_data},, {iteration_idx},, {island_idx},, {is_debug}")
        solution_dir = Path(self.experiments_dir) / f"solutions" / f"iteration_{iteration_idx}_{island_idx}"
        
        solution_dir.mkdir(parents=True, exist_ok=True)
        # print(solutions_dir)

        appendix = ""
        if is_debug:
            appendix = f"_debug"
        success = False
        for key, value in response_data.items():
            
            if key == ResponseEnum.SCORES.value:
                file_ending = ".json"
                value = json.dumps(value)

            elif key == ResponseEnum.CORE_RESPONSE.value:
                file_ending = ".py"
                success = True
            elif "prompt" in key:
                file_ending = ".json"
                value = json.dumps(value)
            else:
                file_ending = ".txt"
            
            file_name = key+appendix+file_ending

            if not os.path.exists(file_name):
                # Save the core response clean code
                with open(solution_dir / file_name, 'w', encoding='utf-8') as f:
                    f.write(value)

        return solution_dir, success

    def _run_eval(self, new_path):
        # call subprocess
        eval_command = ["python", self.evaluate_path]

        # Add reference if set
        if self.reference_path:
            eval_command += ["--reference", self.reference_path]

        # Add solution new_path
        eval_command += ["--solution", new_path]

        # Run the subprocess
        try:
            result = subprocess.run(eval_command, check=True, capture_output=True, text=True)
            # Capture stdout and stderr
            stdout = result.stdout
            stderr = result.stderr
        except Exception as e:
            print("Calling Eval failed.")
            print(e)
        # Check if the evaluation was successful
        if result.returncode == 0:
            success = True
        else:
            success = False

        return extract_eval_results(stdout), stdout, stderr, success

        

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
