"""Hard budget caps for Ada (cost / steps / wall-clock / tokens).

Trigger ``BudgetExceeded`` from inside the agent loop when any configured
limit is hit, so the loop can finish gracefully (write summary, close MCP
servers, etc.) rather than abort mid-write.

All limits are env-tunable and default to "off" (None = unlimited) so
existing behaviour is preserved when the user sets nothing.

Env vars
--------
* ``ADA_MAX_COST``       — total USD across worker + planner.
* ``ADA_MAX_STEPS_HARD`` — secondary step cap (existing ``--steps`` is
  still authoritative; this catches programmatic callers that miss it).
* ``ADA_MAX_TOKENS``     — combined prompt + completion tokens.
* ``ADA_MAX_SECONDS``    — wall-clock since :py:meth:`Budget.start`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


class BudgetExceeded(Exception):
    """Raised when a configured cap (cost / time / tokens / steps) is hit."""

    def __init__(self, kind: str, limit: float, observed: float) -> None:
        super().__init__(f"{kind} budget exceeded: {observed} > {limit}")
        self.kind = kind
        self.limit = limit
        self.observed = observed


def _env_float(name: str) -> float | None:
    """Parse env var as float; return None when unset or malformed."""
    v = os.getenv(name)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    """Parse env var as int; return None when unset or malformed."""
    v = os.getenv(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


@dataclass
class Budget:
    """Soft container for the configured caps + wall-clock anchor."""

    max_cost_usd: float | None = None
    max_steps: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    _t0: float = 0.0

    @classmethod
    def from_env(cls) -> "Budget":
        """Build a Budget from env vars (any unset → no cap on that axis)."""
        return cls(
            max_cost_usd=_env_float("ADA_MAX_COST"),
            max_steps=_env_int("ADA_MAX_STEPS_HARD"),
            max_tokens=_env_int("ADA_MAX_TOKENS"),
            max_seconds=_env_float("ADA_MAX_SECONDS"),
        )

    def start(self) -> None:
        """Anchor the wall-clock origin used by ``check(elapsed=...)``."""
        self._t0 = time.monotonic()

    def check(
        self,
        cost_usd: float = 0.0,
        steps: int = 0,
        tokens: int = 0,
    ) -> None:
        """Raise :class:`BudgetExceeded` for the first cap that's tripped.

        Caller is responsible for accumulating the observables; we only
        compare against the configured limits.
        """
        if self.max_cost_usd is not None and cost_usd > self.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.max_cost_usd, cost_usd)
        if self.max_tokens is not None and tokens > self.max_tokens:
            raise BudgetExceeded("tokens", float(self.max_tokens), float(tokens))
        if self.max_steps is not None and steps > self.max_steps:
            raise BudgetExceeded("steps", float(self.max_steps), float(steps))
        if self.max_seconds is not None and self._t0:
            elapsed = time.monotonic() - self._t0
            if elapsed > self.max_seconds:
                raise BudgetExceeded("seconds", self.max_seconds, elapsed)

    def describe(self) -> dict[str, Any]:
        """Render the active caps for logging / observability."""
        return {
            "cost_usd": self.max_cost_usd,
            "steps": self.max_steps,
            "tokens": self.max_tokens,
            "seconds": self.max_seconds,
        }
