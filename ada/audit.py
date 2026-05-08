"""Structured per-step audit log for Ada.

Each entry is one JSON line in ``.ada/audit.jsonl`` with the schema::

    {"ts": <iso8601>, "step": <int>, "tool": <str>, "args_brief": <str>,
     "ok": <bool>, "duration_ms": <int>, "result_brief": <str>,
     "error": <str|null>}

Designed to be cheap to write (one ``open(..., "a")`` per entry, no
buffering) and trivial to grep / replay.

The companion :py:class:`Replay` utility loads the file back into a list of
dicts so failed runs can be inspected after the fact (``ada-replay <id>``).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Per-result brief preview cap; large bodies (tool stdout, file contents)
# are truncated to keep the audit file usable in plain text editors.
_PREVIEW_CHARS = 240


class AuditLog:
    """Append-only JSONL log of every dispatched tool call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Default-on; opt-out via ADA_AUDIT=0.  Cheap when enabled
        # (one append per tool call, no flush).
        self.enabled = os.getenv("ADA_AUDIT", "1").lower() not in (
            "0", "false", "no", "off",
        )

    def record(
        self,
        step: int,
        tool: str,
        args_brief: str,
        result: str,
        ok: bool,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        """Persist one entry. No-op when :attr:`enabled` is False."""
        if not self.enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": step,
            "tool": tool,
            "args_brief": args_brief[:_PREVIEW_CHARS],
            "ok": ok,
            "duration_ms": duration_ms,
            "result_brief": _trim(result),
            "error": error,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _trim(text: str) -> str:
    """Truncate *text* with a marker so previews stay greppable."""
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS] + f"... [+{len(text) - _PREVIEW_CHARS} chars]"


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read an audit file into a list of dicts (skips malformed lines)."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def summarise(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-tool counts/duration/error totals from a loaded log.

    Useful for ``ada-replay --summary`` and post-mortems on long runs.
    """
    by_tool: dict[str, dict[str, Any]] = {}
    total = 0
    errors = 0
    total_ms = 0
    for e in entries:
        total += 1
        total_ms += int(e.get("duration_ms") or 0)
        if not e.get("ok", True):
            errors += 1
        t = e.get("tool", "?")
        slot = by_tool.setdefault(t, {"calls": 0, "errors": 0, "ms": 0})
        slot["calls"] += 1
        slot["ms"] += int(e.get("duration_ms") or 0)
        if not e.get("ok", True):
            slot["errors"] += 1
    return {
        "total_calls": total,
        "errors": errors,
        "total_ms": total_ms,
        "by_tool": dict(sorted(
            by_tool.items(), key=lambda kv: kv[1]["ms"], reverse=True
        )),
    }
