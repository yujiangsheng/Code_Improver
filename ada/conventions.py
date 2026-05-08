"""Auto-load repo-level conventions into Ada's first user message.

Mirrors how Claude Code reads ``CLAUDE.md`` and Cursor reads ``.cursor/rules``:
on startup we scan the project root for any of a small whitelist of
convention files, concatenate their contents, and inject them into the
kickoff prompt so the agent sees them every run without the user having to
restate them.

Loaded text is **sanitized** before injection (control chars stripped,
fence markers neutralised) so a malicious or sloppy convention file cannot
trivially escape its section and rewrite Ada's system prompt.

The list is deliberately short — repos that need richer rules can drop a
single ``AGENTS.md`` summarising their conventions.
"""
from __future__ import annotations

import re
from pathlib import Path


_CONVENTION_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    ".ada/CONVENTIONS.md",
    ".ada/conventions.md",
    ".github/copilot-instructions.md",
)

_PER_FILE_CAP = 8000          # bytes
_TOTAL_CAP = 16000            # bytes across all files combined

# Strip ASCII control chars (except \n and \t) — they have no business in
# a markdown convention file and can be used to confuse log viewers /
# downstream renderers.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Neutralise our own boundary marker style so a convention file can't fake
# the closing fence and continue with text that looks like a top-level
# instruction to the LLM.
_FENCE_RE = re.compile(r"^={3,}\s*(end|begin)\s+conventions?\b.*$",
                       re.IGNORECASE | re.MULTILINE)


def _sanitize(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = _FENCE_RE.sub("(fence-marker neutralised)", text)
    return text


def load_conventions(root: Path) -> str:
    """Return a sanitized markdown blob of every detected convention file."""
    root = Path(root)
    chunks: list[str] = []
    used = 0
    for rel in _CONVENTION_FILES:
        p = root / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _sanitize(text).strip()
        if not text:
            continue
        if len(text) > _PER_FILE_CAP:
            text = text[:_PER_FILE_CAP] + "\n…[truncated]"
        chunks.append(f"--- {rel} ---\n{text}")
        used += len(text)
        if used >= _TOTAL_CAP:
            break
    if not chunks:
        return ""
    return "\n\n".join(chunks)


def detected_files(root: Path) -> list[str]:
    """List the convention files that exist (for logging)."""
    root = Path(root)
    return [rel for rel in _CONVENTION_FILES if (root / rel).is_file()]
