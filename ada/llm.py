"""LLM client wrapper — uses any OpenAI-compatible Chat Completions API.

Two-tier model routing
----------------------
Ada uses two separate model roles to balance cost and quality:

  Worker  (``LLM``)     — hot path; called every step of the tool loop.
                          Should be fast and code-capable (e.g. qwen3.5:9b).

  Planner (``Planner``) — cold path; invoked only when the worker calls the
                          ``consult_planner`` tool.  Can be a larger / smarter
                          model (e.g. qwen3.5:27b, qwen3-coder:30b, gpt-4o).

Configuration priority (highest → lowest)
-----------------------------------------
1. ``model`` / ``planner_model`` constructor argument
2. ``ADA_WORKER_MODEL`` / ``ADA_PLANNER_MODEL`` env vars
3. ``ADA_MODEL`` env var (shared fallback for both tiers)
4. Hardcoded defaults (``DEFAULT_WORKER_MODEL`` / ``DEFAULT_PLANNER_MODEL``)

Backend
-------
Any OpenAI-compatible endpoint works.  Set:
  ``OPENAI_BASE_URL``  — e.g. ``https://api.openai.com/v1``
  ``OPENAI_API_KEY``   — your API key (Ollama accepts any non-empty string)
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

# ── defaults ─────────────────────────────────────────────────────────────────
# Target a local Ollama daemon so Ada works out-of-the-box without an API key.
# Run `ollama serve` and `ollama pull <model>` to prepare.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"          # Ollama ignores the value; SDK requires one
DEFAULT_WORKER_MODEL  = "qwen3.5:9b"
DEFAULT_PLANNER_MODEL = "qwen3.5:27b"


def _client(api_key: str | None, base_url: str | None) -> OpenAI:
    """Build a shared OpenAI SDK client, resolving config from env as needed."""
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
    )


class LLM:
    """Worker model — drives the main agent tool-calling loop.

    Kept at low temperature (0.2) so output is deterministic and predictable.
    Tool choice is always ``auto``; the model decides when to call a tool
    and when to generate plain text.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Priority: argument → ADA_WORKER_MODEL → ADA_MODEL → built-in default
        self.model = (
            model
            or os.getenv("ADA_WORKER_MODEL")
            or os.getenv("ADA_MODEL")
            or DEFAULT_WORKER_MODEL
        )
        self.client = _client(api_key, base_url)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.2,
    ) -> Any:
        """Send a tool-enabled chat completion request.

        Parameters
        ----------
        messages:
            Full conversation history in OpenAI format.
        tools:
            List of function-calling JSON schemas (``TOOL_SCHEMAS``).
        temperature:
            Sampling temperature.  0.2 gives near-deterministic tool calls.

        Returns
        -------
        ChatCompletion
            Raw OpenAI SDK response object.
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )


class Planner:
    """Planner model — strong one-shot advisor invoked via ``consult_planner``.

    Runs without tools.  The worker passes a detailed prompt; the planner
    responds with prioritised, actionable guidance and returns control.
    Slightly higher temperature (0.3) allows more creative suggestions.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Priority: argument → ADA_PLANNER_MODEL → ADA_MODEL → built-in default
        self.model = (
            model
            or os.getenv("ADA_PLANNER_MODEL")
            or os.getenv("ADA_MODEL")
            or DEFAULT_PLANNER_MODEL
        )
        self.client = _client(api_key, base_url)

    def advise(self, prompt: str, temperature: float = 0.3) -> str:
        """Ask the planner a single question; return its text response.

        Parameters
        ----------
        prompt:
            Full prompt string — typically built by ``Tools.consult_planner``.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            Planner's advice text (stripped of leading/trailing whitespace).
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
