from optimizer_agent import utils
from optimizer_agent import prompt_utils

def extract_final_code_response(response):
    """Extract the final code response from a LLM response string.

    Args:
        response (str): The full response from the LLM.

    Returns:
        str: The extracted code block, or the original response if no code block is found.
    """
    # Look for the start of a code block, usually marked by triple backticks
    start_marker = "```"
    code_start = response.find(start_marker)

    # If no code block is found, return the original response
    if code_start == -1:
        return response

    # Find the end of the code block
    # The end marker is just the triple backticks, not including the language identifier
    code_end = response.find(start_marker, code_start + len(start_marker))

    # If no closing backticks are found, return from the start (including language identifier if present) to the end
    if code_end == -1:
        return response[code_start + len(start_marker):]

    # Otherwise, extract the code between the two markers,
    # skipping any language identifier that may follow the opening backticks
    code_content_start = code_start + len(start_marker)
    # Find the position after any language identifier (e.g., "python", "c++", etc.)
    # by looking for the first newline after the opening backticks
    first_newline = response.find("\n", code_content_start)
    if first_newline != -1 and first_newline < code_end:
        code_content_start = first_newline + 1
    return response[code_content_start:code_end]

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
            messages_strat = prompt_utils.create_prompt(prompt_type="strategy", prompt_data=prompt_data, prompt_config=self.prompt_config)
            
            response_strat = self.server.call_server(messages_strat)[0]
            
            out_dict.update({"strategy_prompt" : messages_strat})
            out_dict.update({"strategy_response" : response_strat})

            prompt_data.update({"strategy" : response_strat})
        
        # ###########
        # Core
        # ###########
        messages_core = prompt_utils.create_prompt(prompt_type="core", prompt_data=prompt_data, prompt_config=self.prompt_config)
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
            messages_feed = prompt_utils.create_prompt(prompt_type="feedback", prompt_data=prompt_data, prompt_config=self.prompt_config)
            
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