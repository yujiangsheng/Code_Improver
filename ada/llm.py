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
import random
import time
from typing import Any, Callable

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

# Exception classes considered transient and worth retrying.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

# ── defaults ─────────────────────────────────────────────────────────────────
# Target a local Ollama daemon so Ada works out-of-the-box without an API key.
# Run `ollama serve` and `ollama pull <model>` to prepare.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"          # Ollama ignores the value; SDK requires one
DEFAULT_WORKER_MODEL  = "qwen3.5:9b"
DEFAULT_PLANNER_MODEL = "qwen3.5:27b"
DEFAULT_REQUEST_TIMEOUT = 600.0
DEFAULT_MAX_RETRIES = 2

# ── provider profiles ────────────────────────────────────────────────────────
# Activated via ADA_PROFILE env var (or the --profile CLI flag).  A profile
# only sets defaults; explicit env vars / CLI args still win.  The api_key
# field names the env var to read (we never hardcode keys).
PROFILES: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OPENAI_API_KEY",  # any string works
        "worker": "qwen3.5:9b",
        "planner": "qwen3.5:27b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "worker": "gpt-4o-mini",
        "planner": "gpt-4o",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "worker": "deepseek-chat",
        "planner": "deepseek-reasoner",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "worker": "moonshot-v1-32k",
        "planner": "moonshot-v1-128k",
    },
    "anthropic": {
        # Anthropic exposes an OpenAI-compatible endpoint at /v1/.
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "worker": "claude-sonnet-4-5",
        "planner": "claude-opus-4-5",
    },
}


def _apply_profile() -> None:
    """Populate env defaults from ``ADA_PROFILE`` (does NOT override existing env)."""
    name = (os.getenv("ADA_PROFILE") or "").strip().lower()
    if not name:
        return
    prof = PROFILES.get(name)
    if prof is None:
        return  # silently ignore unknown profile names
    os.environ.setdefault("OPENAI_BASE_URL", prof["base_url"])
    key_env = prof["api_key_env"]
    if os.getenv(key_env):
        os.environ.setdefault("OPENAI_API_KEY", os.environ[key_env])
    os.environ.setdefault("ADA_WORKER_MODEL", prof["worker"])
    os.environ.setdefault("ADA_PLANNER_MODEL", prof["planner"])


# Apply at import time so downstream env reads see profile defaults.
_apply_profile()


def _env_float(name: str, default: float) -> float:
    """Read *name* from env as float, with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read *name* from env as int, with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _retry_call(fn: Callable[[], Any]) -> Any:
    """Invoke *fn* with exponential backoff on transient API errors.

    Reads ``ADA_LLM_RETRY_ATTEMPTS`` (default 5) and
    ``ADA_LLM_RETRY_BASE`` (default 1.0s).  Backoff is
    ``base * 2**attempt`` capped at 32s, plus 0-25% jitter.
    Non-retryable exceptions propagate immediately.
    """
    attempts = max(1, _env_int("ADA_LLM_RETRY_ATTEMPTS", 5))
    base = max(0.1, _env_float("ADA_LLM_RETRY_BASE", 1.0))
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except _RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(32.0, base * (2 ** attempt))
            delay += delay * random.uniform(0.0, 0.25)
            time.sleep(delay)
    assert last_exc is not None  # for type-checkers
    raise last_exc


def _client(api_key: str | None, base_url: str | None) -> OpenAI:
    """Build a shared OpenAI SDK client, resolving config from env as needed."""
    timeout = _env_float("ADA_LLM_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
    retries = _env_int("ADA_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        timeout=timeout,
        max_retries=retries,
    )


class LLM:
    """Worker model — drives the main agent tool-calling loop.

    Kept at low temperature (0.2) so output is deterministic and predictable.
    Tool choice is always ``auto``; the model decides when to call a tool
    and when to generate plain text.

    Token accounting
    ----------------
    Each successful call accumulates ``self.usage`` (prompt / completion /
    total tokens and request count) so callers can report cost and compare
    models head-to-head.  Read with :py:meth:`usage_summary`.
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
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "requests": 0}

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
        resp = _retry_call(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        ))
        self._record(resp)
        return resp

    def _record(self, resp: Any) -> None:
        """Accumulate token counters from an OpenAI ChatCompletion response."""
        self.usage["requests"] += 1
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        self.usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

    def usage_summary(self) -> dict:
        """Return a copy of the running usage counters (safe to log)."""
        return dict(self.usage, model=self.model)


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
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "requests": 0}

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
        resp = _retry_call(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        ))
        self.usage["requests"] += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0
        return (resp.choices[0].message.content or "").strip()

    def usage_summary(self) -> dict:
        """Return a copy of the running usage counters (safe to log)."""
        return dict(self.usage, model=self.model)
