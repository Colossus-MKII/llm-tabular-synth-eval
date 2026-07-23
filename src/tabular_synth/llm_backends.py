from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleChatConfig:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 120
    retries: int = 3
    system_prompt: str = (
        "You generate synthetic tabular data. Return only machine-parseable CSV rows."
    )


class OpenAICompatibleChatLLM:
    """Callable LLM adapter for OpenAI-compatible chat completion servers.

    Works with services that expose `/v1/chat/completions`, including many
    vLLM and llama.cpp server configurations. For local servers that do not
    check auth, leave the API key environment variable unset.
    """

    def __init__(self, config: OpenAICompatibleChatConfig):
        self.config = config

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self._chat_completions_url()
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"LLM request failed after {self.config.retries} attempts") from last_error

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"
