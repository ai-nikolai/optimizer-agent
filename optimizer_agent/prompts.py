DEFAULT_PROMPT_CONFIG={
    "core":{
        "system": {
            "prompts" : {
                "system": "You are an expert AI scientist."
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is to optimize/improve the given task",
                "implementation" : "## Current Implementation: {current_implementation}",
                "strategy" : "## Strategy: Follow the following strategy {strategy}",
                "feedback" : "## Feedback: Here is additional feedback {feedback}"
            },
            "order" : ["task","implementation","strategy","feedback"]
        }
    },
    "debug":{
        "system": {
            "prompts" :{
                "system" : "You are an expert AI scientist and focus on debugging."
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is to debug the current solution",
                "implementation":"## Current Implementation: {current_implementation}",
                "strategy" : "## Strategy: {strategy}",
                "feedback" : "## Feedback: {feedback}"
            },
            "order" : ["task","implementation","strategy","feedback"]
        }
    },
    "strategy_core":{
        "system": {
            "prompts" :{
                "system" : "You are an expert AI scientist and focus on generating the idea and plan"
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is to create a implementation plan for improving.",
                "implementation": "## Current Implementation: {current_implementation}",
                "result": "## Results: The previous result was: {result}",
                "reference_result": "##Reference result: {reference_result}"
            },
            "order" : ["task","implementation","result","reference_result"]
        }
    },
    "strategy_debug":{
        "system": {
            "prompts" :{
                "system" : "You are an expert AI scientist and focus on debugging and specifically on generating the idea and plan"
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is to create a implementation plan for improving.",
                "implementation": "## Current Implementation: {current_implementation}",
                "error": "## Error: {error}"
            },
            "order" : ["task","implementation","error"]
        }
    },
    "feedback_core":{
        "system": {
            "prompts" :{
                "system":"You are an expert AI scientist and focus on reviewing previous solutions and generating feedback"
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is provide feedback on the implementation",
                "implementation": "## Current Implementation: {current_implementation}",
                "result": "## Results: The previous result was: {result}"
            },
            "order" : ["task","implementation","result"]
        }
    },
    "feedback_debug":{
        "system": {
            "prompts" :{
                "system":"You are an exper AI scientist and focus debugging and specifically on reviewing previous solutions and generating feedback"
            },
            "order" : ["system"]
        },
        "user": {
            "prompts" : {
                "task" : "## Task: Your task is to improve the speed of the CUDA implementation",
                "implementation": "## Current Implementation: {current_implementation}",
                "error": "## Error: {error}",
                "reference" : "## Reference: Here is the reference implementation {reference}"
            },
            "order" : ["task","implementation","error","reference"]
        }
    }
}


def create_prompt(prompt_type="core", key_info={}, prompt_config=DEFAULT_PROMPT_CONFIG,):
    """
    Takes a prompt config and automatically constructs the final messages object for the current prompt_type. If a key is missing it just skips that line.
    """
    # Get the prompt configuration for the specified type
    cfg = prompt_config.get(prompt_type)
    if cfg is None:
        raise ValueError(f"Prompt type '{prompt_type}' not found in config.")

    messages = []

    # Process system prompts
    system_cfg = cfg.get("system", {})
    system_messages = []
    for key in system_cfg.get("order", []):
        prompt = system_cfg.get("prompts", {}).get(key)
        if prompt:
            # Format the prompt with key_info if available
            try:
                formatted_prompt = prompt.format(**key_info)
            except KeyError:
                # Skip if a key is missing
                continue
            system_messages.append(formatted_prompt)

    if system_messages:
        messages.append({"role": "system", "content": "\n".join(system_messages)})

    # Process user prompts
    user_cfg = cfg.get("user", {})
    user_messages = []
    for key in user_cfg.get("order", []):
        prompt = user_cfg.get("prompts", {}).get(key)
        if prompt:
            # Format the prompt with key_info if available
            try:
                formatted_prompt = prompt.format(**key_info)
            except KeyError:
                # Skip if a key is missing
                continue
            user_messages.append(formatted_prompt)

    if user_messages:
        messages.append({"role": "user", "content": "\n".join(user_messages)})

    return messages




def compaction_prompt(messages):
    """
    Takes a messages list and returns a compact prompt string summarizing the conversation.
    """
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt = f"""
Please analyze the following conversation and output a concise summary covering:
1. Key topics discussed
2. Any decisions made
3. Important information or facts provided
4. Action items or next steps (if any)

Keep the summary to 2-3 sentences.

Conversation:
{conversation}
"""
    return prompt.strip()

if __name__=="__main__":
    print(create_prompt())