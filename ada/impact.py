"""Impact analysis & diff stats for change-aware testing.

* :func:`changed_test_targets` — given a list of changed source files,
  return the test files that import (directly or transitively) any
  module under one of them.  Uses a textual scan of ``import`` /
  ``from X import`` lines — no AST, no LSP, no third-party deps.

* :func:`diff_stats` — summarise a unified-diff blob into per-file
  added/removed line counts plus a risk verdict ("safe" / "review" /
  "risky") based on hunk size.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_SKIP_DIRS = {".git", ".ada", "__pycache__", ".venv", "node_modules", ".tox", "dist", "build"}
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))",
    re.MULTILINE,
)


def changed_test_targets(root: Path, changed_files: list[str]) -> dict:
    """Map *changed_files* (under *root*) to candidate test files.

    Returns ``{"tests": [...], "modules": [...]}``.  ``modules`` is the
    derived dotted-module list for each changed ``.py`` file; ``tests``
    is the union of test files whose ``import``/``from`` lines mention
    any of those modules (or the bare basename for safety).
    """
    root = Path(root).resolve()
    modules: list[str] = []
    basenames: set[str] = set()
    for rel in changed_files:
        if not rel.endswith(".py"):
            continue
        p = (root / rel).resolve()
        # Drop the .py and convert path separators to dots.
        try:
            rel_path = p.relative_to(root).with_suffix("")
        except ValueError:
            continue
        parts = list(rel_path.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        modules.append(".".join(parts))
        basenames.add(parts[-1])

    if not modules:
        return {"tests": [], "modules": []}

    needles = set(modules) | basenames
    tests: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            if not (fn.startswith("test_") or fn.endswith("_test.py")):
                continue
            fp = Path(dirpath) / fn
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hit = False
            for m in _IMPORT_RE.finditer(text):
                imported = m.group(1) or (m.group(2) or "").replace(" ", "")
                if not imported:
                    continue
                # imported can be "a, b" — check each.
                for token in imported.split(","):
                    token = token.split(" as ")[0].strip()
                    if not token:
                        continue
                    if token in needles or token.split(".")[-1] in needles:
                        hit = True
                        break
                    # Module-prefix match ("pkg.sub" matches changed "pkg.sub.x").
                    for mod in modules:
                        if mod == token or mod.startswith(token + ".") or token.startswith(mod + "."):
                            hit = True
                            break
                if hit:
                    break
            if hit:
                tests.append(str(fp.relative_to(root)))
    return {"tests": sorted(set(tests)), "modules": sorted(set(modules))}


def diff_stats(diff_text: str) -> dict:
    """Per-file added/removed counts plus a coarse risk verdict.

    Verdict thresholds:
      * "safe"   — total churn <= 30 lines
      * "review" — 31-200 lines
      * "risky"  — > 200 lines or any single file > 100 lines
    """
    files: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].strip()
            files.setdefault(current, {"added": 0, "removed": 0})
            continue
        if line.startswith("--- a/"):
            continue
        if current is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            files[current]["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            files[current]["removed"] += 1

    total = sum(f["added"] + f["removed"] for f in files.values())
    biggest = max((f["added"] + f["removed"] for f in files.values()), default=0)
    if total > 200 or biggest > 100:
        verdict = "risky"
    elif total > 30:
        verdict = "review"
    else:
        verdict = "safe"
    return {
        "files": [{"path": p, **counts} for p, counts in files.items()],
        "total_changed_lines": total,
        "verdict": verdict,
    }
