"""Thin wrappers around an OpenAI-compatible chat endpoint.

The agent talks to a model server (sglang / vllm / openai) through a single,
tiny interface: :meth:`BaseServer.call`. Nothing else in the code touches the
HTTP layer, which makes everything above it trivial to mock and test.
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Tuple


class BaseServer:
    """The interface the agent uses to call a model."""

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        model: str = "default",
    ) -> Tuple[List[str], List[Any]]:
        raise NotImplementedError


class LLMServer(BaseServer):
    """Query an OpenAI-compatible server via ``/v1/chat/completions``.

    Works with local (sglang / vllm) and hosted servers. Authentication is
    optional. Failures never propagate -- on any network/parse error the call
    returns an empty string so the loop can retry.
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
    ) -> Tuple[List[str], List[Any]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            message = resp.json().get("choices", [{}])[0].get("message", {})
            return [message.get("content", "")], message.get("tool_calls", [])
        except Exception as exc:  # keep failures recoverable in the loop
            print(f"Error calling LLM server: {exc}")
            return [""], []


class FakeLLMServer(BaseServer):
    """Deterministic-ish mock used for ``--test`` runs.

    Never touches the network. Returns a short random string so the whole
    control flow (prompt building -> generation -> evaluation -> feedback) is
    exercised on any machine, with or without a model server.
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
    ) -> Tuple[List[str], List[Any]]:
        length = random.randint(150, 500)
        payload = "".join(random.choices(string.ascii_letters + string.digits, k=length))
        return [payload], []
