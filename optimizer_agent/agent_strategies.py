from optimizer_agent import utils
from optimizer_agent import utils_prompt

from optimizer_agent.utils import (
    extract_final_code_response,
)

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
            
            response_strat = self.server.call_server(messages_strat)[0]
            
            out_dict.update({"strategy_prompt" : messages_strat})
            out_dict.update({"strategy_response" : response_strat})

            prompt_data.update({"strategy" : response_strat})
        
        # ###########
        # Core
        # ###########
        messages_core = utils_prompt.create_prompt(prompt_type="core", prompt_data=prompt_data, prompt_config=self.prompt_config)
        response_core = self.server.call_server(messages_core)[0]
        response_core_clean = extract_final_code_response(response_core)
        out_dict.update({"core_prompt" : messages_core})
        out_dict.update({"core_response" : response_core})
        out_dict.update({"core_response_clean" : response_core_clean})
        
        prompt_data.update({"new_implementation" : response_core_clean})           
        
        # ###########
        # Feedback
        # ###########
        if self.use_feedback:
            messages_feed = utils_prompt.create_prompt(prompt_type="feedback", prompt_data=prompt_data, prompt_config=self.prompt_config)
            
            response_feed = self.server.call_server(messages_feed)[0]
            
            out_dict.update({"feedback_prompt" : messages_feed})
            out_dict.update({"feedback_response" : response_feed})

            prompt_data.update({"feedback" : response_strat})

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