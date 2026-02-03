#!/usr/bin/env python3
"""Minimal LLM client for local HTTP chat-completions endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class LLMConfig:
    """Configuration describing how to talk to the LLM backend."""

    provider: str
    model: str
    api_base: str
    api_key: str = ""
    device: str = "auto"
    max_new_tokens: Optional[int] = None
    temperature: float = 0.6


class LLMClient:
    """Simple HTTP client that mirrors OpenAI's chat completions schema."""

    def __init__(self, config: LLMConfig):
        if not config.api_base:
            if config.provider == "openai":
                config.api_base = "https://api.openai.com/v1"
            else:
                raise ValueError(
                    "LLM api_base is not set. Provide --llm-api-base for non-OpenAI providers."
                )
        self.config = config
        self._session = requests.Session()
        if config.api_key:
            self._session.headers["Authorization"] = f"Bearer {config.api_key}"
        self._session.headers.setdefault("Content-Type", "application/json")

    def _request_payload(self, prompt: str, max_tokens: Optional[int]) -> Dict[str, Any]:
        """Format payload for chat completions style endpoints."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not self.config.model.startswith("gpt-5"):
            payload["temperature"] = self.config.temperature
        token_limit = max_tokens if max_tokens is not None else self.config.max_new_tokens
        if token_limit is not None:
            if self.config.model.startswith("gpt-5"):
                payload["max_completion_tokens"] = token_limit
            else:
                payload["max_tokens"] = token_limit
        return payload

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Send a chat completion request and return the assistant content.
        
        Raises RuntimeError if the backend responds with an error or
        deviates from the expected schema.
        """
        payload = self._request_payload(prompt, max_tokens)
        response = self._session.post(self.config.api_base.rstrip("/") + "/chat/completions", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"LLM request failed ({response.status_code}): {response.text}")
        data = response.json()
        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                raise RuntimeError(f"LLM returned empty content: {data}")
            return content
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response schema: {data}") from exc
