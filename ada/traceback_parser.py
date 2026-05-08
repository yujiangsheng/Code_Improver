"""Parse pytest / unittest tracebacks → list of (file, line, symbol).

Used by Ada to auto-pull the relevant source into context when tests
fail, instead of waiting for the model to re-issue ``read_file`` calls
for each frame.

We deliberately accept noisy mixed input (rich's panels, stderr lines,
ANSI escapes) and pull out anything that looks like a Python frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Matches the canonical "  File "...", line N, in <symbol>" frame line
# emitted by both unittest and pytest --tb=long/short.
_FRAME_RE = re.compile(
    r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<sym>[^\s]+)'
)
# pytest --tb=line / --tb=auto sometimes uses "path:line:" style.
_SHORT_RE = re.compile(
    r'^(?P<file>[^\s:]+\.py):(?P<line>\d+):\s', re.MULTILINE
)


@dataclass(frozen=True)
class Frame:
    file: str
    line: int
    symbol: str

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "symbol": self.symbol}


def parse(text: str) -> list[Frame]:
    """Extract frames from a traceback blob.  Stable order, deduped.

    Both styles are merged so we get both the rich frame info (with
    symbol) and the bare path:line hits from pytest's short tracebacks.
    """
    seen: set[tuple[str, int, str]] = set()
    out: list[Frame] = []
    for m in _FRAME_RE.finditer(text):
        key = (m.group("file"), int(m.group("line")), m.group("sym"))
        if key in seen:
            continue
        seen.add(key)
        out.append(Frame(file=key[0], line=key[1], symbol=key[2]))
    for m in _SHORT_RE.finditer(text):
        key = (m.group("file"), int(m.group("line")), "?")
        if any(f.file == key[0] and f.line == key[1] for f in out):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(Frame(file=key[0], line=key[1], symbol="?"))
    return out


def in_workspace(frames: Iterable[Frame], root: Path) -> list[Frame]:
    """Filter to frames whose file lives under *root*.

    Skips third-party / stdlib frames (those typically dominate noisy
    tracebacks but are rarely actionable for the agent).
    """
    root = root.resolve()
    out: list[Frame] = []
    for f in frames:
        try:
            p = Path(f.file).resolve()
        except OSError:
            continue
        try:
            p.relative_to(root)
        except ValueError:
            continue
        out.append(Frame(file=str(p.relative_to(root)), line=f.line, symbol=f.symbol))
    return out
