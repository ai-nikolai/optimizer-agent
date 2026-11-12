"""OpenAI-compatible chat + embedding client.

Chat calls retry transient failures with bounded backoff. After retries they
raise :class:`LLMError` instead of silently returning an empty artifact.
"""

import random
import string
import time

import requests


class LLMError(RuntimeError):
    """Raised when the model endpoint cannot produce a usable response."""


class LLMServer:
    """Call an OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        server_url=None,
        api_key=None,
        timeout=600,
        retries=3,
        retry_backoff=1.0,
    ):
        self.server_url = server_url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    def _headers(self, api_key=None):
        headers = {"Content-Type": "application/json"}
        key = api_key if api_key is not None else self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @staticmethod
    def _sleep(attempt, attempts, backoff):
        if attempt < attempts - 1 and backoff:
            time.sleep(backoff * (2**attempt))

    def call_server(
        self,
        messages,
        model="default",
        max_tokens=4096,
        temperature=0.0,
        top_p=None,
        tools=None,
        retries=None,
        retry_backoff=None,
        **params,
    ):
        """Return ``([content], tool_calls)`` or raise :class:`LLMError`."""
        url = f"{self.server_url}/v1/chat/completions"
        attempts = max(1, int(retries if retries is not None else self.retries))
        backoff = (
            self.retry_backoff if retry_backoff is None else max(0.0, float(retry_backoff))
        )

        payload = dict(params)
        payload["model"] = model
        payload["messages"] = messages
        if temperature is not None:
            payload.setdefault("temperature", temperature)
        if max_tokens is not None:
            payload.setdefault("max_tokens", max_tokens)
        if top_p is not None:
            payload.setdefault("top_p", top_p)
        if tools:
            payload.setdefault("tools", tools)
            payload.setdefault("tool_choice", "auto")

        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                choices = response.json().get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                content = message.get("content") or ""
                if content.strip():
                    return [content], message.get("tool_calls") or []
                last_error = "empty response content"
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
            except (ValueError, KeyError, IndexError) as exc:
                last_error = f"could not process response: {exc}"
            self._sleep(attempt, attempts, backoff)

        raise LLMError(f"LLM request failed after {attempts} attempt(s): {last_error}")

    def embed(
        self,
        texts,
        model,
        url=None,
        api_key=None,
        retries=None,
    ):
        """Return embeddings for ``texts`` from an OpenAI-compatible endpoint."""
        base = (url or self.server_url).rstrip("/")
        endpoint = f"{base}/v1/embeddings"
        attempts = max(1, int(retries if retries is not None else self.retries))
        payload = {"model": model, "input": list(texts)}
        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                response = requests.post(
                    endpoint,
                    headers=self._headers(api_key),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json().get("data") or []
                vectors = [item.get("embedding") for item in data]
                if len(vectors) != len(texts) or any(vector is None for vector in vectors):
                    last_error = "embedding response shape mismatch"
                else:
                    return vectors
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
            except (ValueError, KeyError, TypeError) as exc:
                last_error = f"could not process embedding response: {exc}"
            self._sleep(attempt, attempts, self.retry_backoff)
        raise LLMError(
            f"embedding request failed after {attempts} attempt(s): {last_error}"
        )


class FakeLLMServer:
    """Offline server for ``--test``: returns random text, never the network."""

    def __init__(self, *args, **kwargs):
        print("Fake LLM server running (no network calls).")

    def call_server(self, *args, **kwargs):
        length = random.randint(150, 500)
        text = "".join(random.choices(string.ascii_letters + string.digits, k=length))
        return [text], []

    def embed(self, texts, *args, **kwargs):
        return [[float(len(str(text))), 1.0, 0.0] for text in texts]
