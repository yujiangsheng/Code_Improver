"""Token counting helpers for context-budget-aware compaction.

We try to use ``tiktoken`` (works for OpenAI / DeepSeek / most Qwen models —
their tokenizers are close enough for budgeting purposes).  If tiktoken is not
installed, we fall back to a conservative ``len(text) / 3`` estimate so the
compactor still triggers on the right side of the real limit.

Public API
----------
``count_text(s)``       — tokens in a single string
``count_messages(ms)``  — tokens for a list of OpenAI chat messages
                          (includes a small per-message + tool-call overhead)
"""
from __future__ import annotations

import json
from typing import Any

try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                                             # pragma: no cover
    _ENC = None


# Per-message framing overhead (role + separators).  OpenAI's own cookbook
# uses 3–4; we use 4 as a safe round number.
_PER_MESSAGE_OVERHEAD = 4


def count_text(s: str | None) -> int:
    """Return the token count for *s* (``None`` and empty string → 0)."""
    if not s:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(s, disallowed_special=()))
        except Exception:
            pass
    # Fallback: ~3 chars/token is a conservative overestimate for English/code.
    return max(1, len(s) // 3)


def count_messages(messages: list[dict[str, Any]]) -> int:
    """Return total tokens for a list of OpenAI chat messages.

    Counts ``content`` plus any ``tool_calls`` arguments / names so messages
    that contain only a function call (no plain content) are not undercounted.
    """
    total = 0
    for m in messages:
        total += _PER_MESSAGE_OVERHEAD
        c = m.get("content")
        if isinstance(c, str):
            total += count_text(c)
        elif isinstance(c, list):
            # Multimodal content blocks — count each text segment.
            for blk in c:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    total += count_text(blk["text"])
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += count_text(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                total += count_text(args)
            elif args is not None:
                total += count_text(json.dumps(args, ensure_ascii=False))
        if m.get("name"):
            total += count_text(m["name"])
    return total


def has_tokenizer() -> bool:
    """True when a real tokenizer (tiktoken) is in use, False on fallback."""
    return _ENC is not None
