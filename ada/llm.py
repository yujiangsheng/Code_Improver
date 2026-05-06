"""LLM client wrapper — uses any OpenAI-compatible Chat Completions API.

Two-tier setup:
  * worker model — drives the main tool-using loop (fast / cheap)
  * planner model — one-shot advisor invoked via the `consult_planner` tool
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

# Defaults: local Ollama (no API key required).
# Override with env vars / CLI flags as needed.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # Ollama ignores the value but the SDK requires one
DEFAULT_WORKER_MODEL = "qwen3-coder"
DEFAULT_PLANNER_MODEL = "qwen3"


def _client(api_key: str | None, base_url: str | None) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
    )


class LLM:
    """Worker model — drives the main agent loop with tools."""

    def __init__(self, model: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None):
        self.model = (
            model
            or os.getenv("ADA_WORKER_MODEL")
            or os.getenv("ADA_MODEL")
            or DEFAULT_WORKER_MODEL
        )
        self.client = _client(api_key, base_url)

    def chat(self, messages: list[dict], tools: list[dict],
             temperature: float = 0.2) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )


class Planner:
    """Planner model — strong one-shot advisor, no tools."""

    def __init__(self, model: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None):
        self.model = (
            model
            or os.getenv("ADA_PLANNER_MODEL")
            or os.getenv("ADA_MODEL")
            or DEFAULT_PLANNER_MODEL
        )
        self.client = _client(api_key, base_url)

    def advise(self, prompt: str, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
