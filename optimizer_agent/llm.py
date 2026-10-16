"""Thin wrapper around an OpenAI-compatible chat-completions endpoint.

The optimizer agent talks to local model servers (sglang / vllm / openai)
through a single, small interface: :meth:`LLMServer.call`. The rest of the
code never imports ``requests`` directly, which makes it trivial to mock and
to test.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any, Dict, List, Optional, Tuple


class BaseServer:
    """Common interface used by the agent."""

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        model: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[Any]]:
        raise NotImplementedError


class LLMServer(BaseServer):
    """Query an OpenAI-compatible server via the ``/v1/chat/completions`` API.

    Works with both local (sglang / vllm) and hosted servers. Authentication is
    optional: if an API key is supplied an ``Authorization: Bearer`` header is
    sent.
    """

    def __init__(self, server_url: Optional[str] = None, api_key: Optional[str] = None):
        self.server_url = server_url
        self.api_key = api_key
        self.endpoint = f"{server_url}/v1/chat/completions"

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        model: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[Any]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json().get("choices", [{}])[0].get("message", {})
            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])
            return [content], tool_calls
        except Exception as exc:  # keep failures recoverable in the loop
            print(f"Error calling LLM server: {exc}")
            return [""], []

    def call_prompt(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        model: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[Any]]:
        """Convenience wrapper that builds a single-user-message prompt."""
        return self.call(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
            tools=tools,
        )


class FakeLLMServer(BaseServer):
    """Deterministic mock used for smoke tests and ``--test`` runs.

    It never connects to a network and always returns a short random string.
    This keeps the agent's control flow exercised without hardware.
    """

    def __init__(self, *args, **kwargs):
        print("FakeLLMServer running...")

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        model: str = "default",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[Any]]:
        length = random.randint(150, 500)
        payload = "".join(random.choices(string.ascii_letters + string.digits, k=length))
        return [payload], []
