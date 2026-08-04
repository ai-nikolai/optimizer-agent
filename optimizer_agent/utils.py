import requests
import json
import argparse

def extract_eval_results(stdout):
    # Extract the final JSON output after EVAL_MARKER
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
                return evaluation_result
            else:
                print(f"Warning: No JSON found after EVAL_MARKER in stdout")
                print(f"Evaluation output: {evaluation_output[:500]}...")
                return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from evaluation output: {e}")
            print(f"Raw output after marker: {evaluation_output[:500]}...")
            return {}
    else:
        print(f"Warning: EVAL_MARKER '{marker}' not found in stdout")
        print(f"Stdout: {stdout[:500]}...")
        if stderr:
            print(f"Stderr: {stderr[:500]}...")
        return {}

class LLMServer():

    def __init__(self, server_url=None, api_key=None,):
        self.server_url = server_url
        self.api_key = api_key
    
    
    def call_server(self, messages, max_tokens=2048, temperature=0.0, top_p=0.95, model="default", tools=None,):
        """
        Querying an OpenAI-compatible server (local or remote) using the requests library.
        Supports authentication via API key if provided.
        """
        url = f"{self.server_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # "max_tokens": max_tokens,
            # "top_p": top_p
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # TODO change this for multiple responses.
            result = response.json().get("choices", [{}])[0].get("message", {})
            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            return [content], tool_calls

        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenAI Server...: {e}")
            return None, None
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error processing response: {e}")
            return None, None

    def call_server_prompt(self, prompt, max_tokens=1024, temperature=0.0, top_p=0.95, model="default", tools=None):
        """
        OpenAI compatible call, pre-creating the messages object.
        """
        messages = [
                {"role": "user", "content": prompt}
            ]

        return self.call_server(messages, max_tokens, temperature, top_p, model, tools=tools)



class FakeLLMServer():
    def __init__(self,*argv,**kwargs):
        print("Fake Server Running...")
    
    def call_server(self, *argv,**kwargs):
        return ["fake server response"]
    
    def call_server_prompt(self, *argv,**kwargs):
        return ["fake prompt response"]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=30000, help="API port")
    parser.add_argument("--prompt", type=str, default="What is the purpose of life?", help="prompt")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication (optional)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Configuration for local SGLang deployment
    server_url=f"http://{args.host}:{args.port}" 

    print("="*60)
    print("Entering LLM call:")
    print("-"*30)
    prompt = args.prompt

    print(f"Prompt:\n{prompt}")
    print("---")

    server = LLMServer(server_url, args.api_key)
 
    response_text, tool_calls = server.call_server_prompt(prompt)
    print(f"Response:\n{response_text}")
    print("---")
    print(f"Tools:\n{tool_calls}")
    print("---")



    # #################################
    # Test call to DeepSeek endpoint
    # #################################

    # deepseek_server_url = "https://api.deepseek.com"
    # deepseek_model = "deepseek-v4-flash"
    # print("="*60)
    # print("Entering DeepSeek call:")
    # print("-"*30)
    # response_text_ds, tools_ds = call_openai_server_prompt(prompt, server_url=deepseek_server_url, api_key=args.api_key, model=deepseek_model, tools=tools_dict)
    # print(f"DeepSeek Response:\n{response_text_ds}")
    # print("---")
    # print(f"Tools:\n{tools_ds}")
    # print("---")
