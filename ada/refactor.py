"""Cross-file symbol rename: textual occurrences with safety checks.

Not as smart as a true LSP rename (no scope analysis) but good enough for
the common case of "rename this top-level function/class everywhere it
appears" with these guarantees:

* Only matches whole-word identifiers (``\\b`` boundaries).
* Preserves Python imports automatically — ``from m import OldName`` and
  ``import m as OldName`` get rewritten too.
* Skips binary files, ``.git/``, ``.venv/``, build dirs.
* Returns a per-file count so the agent can spot suspicious matches.

Use ``dry_run=True`` to get the would-change list without touching disk.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = {".git", ".ada", "__pycache__", ".venv", "node_modules", ".tox", "dist", "build"}
_TEXT_EXTS = {".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
              ".java", ".kt", ".rb", ".c", ".cc", ".cpp", ".h", ".hpp",
              ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}


@dataclass
class RenameHit:
    path: str
    occurrences: int


def rename_symbol(
    root: Path,
    old: str,
    new: str,
    dry_run: bool = False,
    extensions: list[str] | None = None,
) -> dict:
    """Rename whole-word identifier *old* → *new* across the tree.

    Returns ``{"hits": [...], "files_changed": int, "total_replacements": int}``.
    Refuses obviously-unsafe renames (empty / whitespace / non-identifier).
    """
    if not _is_identifier(old) or not _is_identifier(new):
        return {"error": "old/new must be valid identifiers (\\w+, no spaces)"}
    if old == new:
        return {"error": "old and new are identical"}

    pattern = re.compile(r"\b" + re.escape(old) + r"\b")
    exts = set(extensions) if extensions else _TEXT_EXTS
    hits: list[RenameHit] = []
    total = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            ext = Path(fn).suffix
            if ext not in exts:
                continue
            fp = Path(dirpath) / fn
            try:
                if fp.stat().st_size > 1_000_000:
                    continue
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            count = len(pattern.findall(text))
            if count == 0:
                continue
            total += count
            hits.append(RenameHit(path=str(fp.relative_to(root)), occurrences=count))
            if not dry_run:
                fp.write_text(pattern.sub(new, text), encoding="utf-8")

    return {
        "hits": [{"path": h.path, "occurrences": h.occurrences} for h in hits],
        "files_changed": 0 if dry_run else len(hits),
        "total_replacements": total,
        "dry_run": dry_run,
    }


def _is_identifier(s: str) -> bool:
    """True iff *s* is a non-empty Python-style identifier (single token)."""
    return bool(s) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s))
