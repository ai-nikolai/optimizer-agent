from enum import Enum

from optimizer_agent import utils
from optimizer_agent import utils_prompt

from optimizer_agent.utils import (
    extract_final_code_response,
)

class ResponseEnum(Enum):
    STRATEGY_PROMPT="strategy_prompt"
    STRATEGY_RESPONSE_RAW="strategy_response_raw"

    CORE_PROMPT="core_prompt"
    CORE_RESPONSE_RAW="core_response_raw"
    CORE_RESPONSE="core_response"

    FEEDBACK_PROMPT="feedback_prompt"
    FEEDBACK_RESPONSE_RAW="feedback_response_raw"

    SCORES="scores"
    EVAL_STDOUT="eval_stdout"
    EVAL_STDERR="eval_stdout"

# TODO: FIL can be implemented here by changing the feedback mechanism... (as a special case for the debug agent)
# TODO: DIFF Agent can be maybe implemented here as well...

class CoreAgent():    
    def __init__(self, server, prompt_config=None, use_strategy=False, use_feedback=False):
        self.server = server

        self.prompt_config = prompt_config
        self.use_strategy = use_strategy
        self.use_feedback = use_feedback

    
    def generate_response(self, prompt_data):
        out_dict = {}
        # ###########
        # Strategy
        # ########### 
        if self.use_strategy:
            messages_strat = utils_prompt.create_prompt(prompt_type="strategy", prompt_data=prompt_data, prompt_config=self.prompt_config)
            
            response_strat_list, _ = self.server.call_server(messages_strat)
            response_strat = response_strat_list[0]
            
            out_dict.update({ResponseEnum.STRATEGY_PROMPT.value : messages_strat})
            out_dict.update({ResponseEnum.STRATEGY_RESPONSE_RAW.value : response_strat})

            prompt_data.update({"strategy" : response_strat})
        
        # ###########
        # Core
        # ###########
        messages_core = utils_prompt.create_prompt(prompt_type="core", prompt_data=prompt_data, prompt_config=self.prompt_config)
        response_core_list, _ = self.server.call_server(messages_core)
        response_core = response_core_list[0]

        response_core_clean = extract_final_code_response(response_core)
        out_dict.update({ResponseEnum.CORE_PROMPT.value : messages_core})
        out_dict.update({ResponseEnum.CORE_RESPONSE_RAW.value : response_core})
        out_dict.update({ResponseEnum.CORE_RESPONSE.value : response_core_clean})
        
        prompt_data.update({"new_implementation" : response_core_clean})           
        
        # ###########
        # Feedback
        # ###########
        # TODO: FIL can be implemented here by changing the feedback mechanism...
        if self.use_feedback:
            messages_feed = utils_prompt.create_prompt(prompt_type="feedback", prompt_data=prompt_data, prompt_config=self.prompt_config)
            
            response_feed_list, _ = self.server.call_server(messages_feed)
            response_feed = response_feed_list[0]
            
            out_dict.update({ResponseEnum.FEEDBACK_PROMPT.value : messages_feed})
            out_dict.update({ResponseEnum.FEEDBACK_RESPONSE_RAW.value : response_feed})

            prompt_data.update({"feedback" : response_feed})

        return out_dict, prompt_data     


if __name__ == "__main__":
    from pprint import pprint

    server = utils.FakeLLMServer()
    agent = CoreAgent(server, use_strategy=True, use_feedback=True)

    out, prompt_data = agent.generate_response({})
   
    print("====")
    pprint(out)
    print("----")
    pprint(prompt_data)