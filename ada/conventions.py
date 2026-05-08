"""Auto-load repo-level conventions into Ada's first user message.

Mirrors how Claude Code reads ``CLAUDE.md`` and Cursor reads ``.cursor/rules``:
on startup we scan the project root for any of a small whitelist of
convention files, concatenate their contents, and inject them into the
kickoff prompt so the agent sees them every run without the user having to
restate them.

The list is deliberately short — repos that need richer rules can drop a
single ``AGENTS.md`` summarising their conventions.
"""
from __future__ import annotations

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


def load_conventions(root: Path) -> str:
    """Return a markdown blob of every detected convention file (or empty).

    The returned string is empty when no files are found, which lets the
    caller append it conditionally without leaving a blank section in the
    prompt.
    """
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
        text = text.strip()
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
