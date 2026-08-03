# System prompt defining available tools and compaction behavior
def system_prompt():
    """
    System prompt for agent optimizer...
    """
    prompt = ""
    prompt += "You are an amazing scientist, engineer and discoverer. You think about things deeply, but are also very concise in your thoughts. You will be given a task that you need to optimize."
    return prompt.strip()

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
    print(system_prompt())