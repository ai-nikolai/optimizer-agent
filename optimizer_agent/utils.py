import requests
import json
import argparse


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


def extract_eval_results(stdout):
    # Extract the final JSON output after EVAL_MARKER
    default_eval_results = {'combined_score': -float("inf")}
    marker = "=== Evaluation Result ==="
    if marker in stdout:
        # Get everything after the marker
        evaluation_output = stdout.split(marker)[-1].strip()

        # Try to parse the JSON from the evaluation output
        try:
            # Find the first JSON object in the output
            # This handles cases where there might be additional text
            start_idx = evaluation_output.find('{')
            end_idx = evaluation_output.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = evaluation_output[start_idx:end_idx]
                evaluation_result = json.loads(json_str)
                print(f"Info: Got eval results: {evaluation_result}.")
                return evaluation_result
            else:
                print(f"Warning: No JSON found after EVAL_MARKER in stdout")
                print(f"Evaluation output: {evaluation_output[:500]}...")
                return default_eval_results
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from evaluation output: {e}")
            print(f"Raw output after marker: {evaluation_output[:500]}...")
            return default_eval_results
    else:
        print(f"Warning: EVAL_MARKER '{marker}' not found in stdout")
        print(f"Stdout: {stdout[:500]}...")
        if stderr:
            print(f"Stderr: {stderr[:500]}...")
        return default_eval_results



if __name__=="__main__":
    out_dict = {'combined_score':0.1}
    
    eval_results = f"""random string\nand another
=== Evaluation Result ===
{json.dumps(out_dict)}
"""
    print(eval_results)
    results = extract_eval_results(eval_results)

    print(results)

    print(results==out_dict)
