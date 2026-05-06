"""Workspace manager — owns the ``.ada/`` directory inside the target project.

The workspace is Ada's memory: it persists state between sessions, records
progress, and provides a sandboxed view of the file system so the agent
cannot accidentally modify files outside the target directory.

Artifact files
--------------
All files live under ``<target_dir>/.ada/``.

  profile.md        — high-level project profile written in Phase 0
  mental_model.md   — module map, data-flow, key abstractions (Phase 1)
  baseline.json     — numeric baseline captured before any changes
  backlog.md        — prioritised improvement table updated each cycle
  journal.md        — timestamped log of every iteration
  metrics.csv       — time-series metric rows (timestamp, phase, metric, value)
  questions.md      — open questions for the human operator

Missing files are auto-created with placeholder content on first run so Ada
always has a consistent directory structure to work with.

Path sandbox
------------
``Workspace.resolve()`` converts any relative or absolute path to an absolute
one and verifies it is inside ``target_dir``.  This is the single choke-point
that prevents path-traversal attacks (e.g. ``../../etc/passwd``).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Workspace:
    """Manages Ada's persistent ``.ada/`` working directory.

    Parameters
    ----------
    target_dir:
        Absolute or relative path to the project Ada is improving.
        Must already exist as a directory.

    Raises
    ------
    NotADirectoryError
        If ``target_dir`` does not exist or is not a directory.
    """

    #: All recognised artifact filenames.  Only these may be written via
    #: ``update_artifact`` to avoid the agent creating arbitrary files.
    ARTIFACTS = [
        "profile.md",
        "mental_model.md",
        "baseline.json",
        "backlog.md",
        "journal.md",
        "metrics.csv",
        "questions.md",
    ]

    def __init__(self, target_dir: str | Path) -> None:
        self.target_dir = Path(target_dir).expanduser().resolve()
        if not self.target_dir.is_dir():
            raise NotADirectoryError(f"Target directory not found: {self.target_dir}")
        self.ada_dir = self.target_dir / ".ada"
        self.ada_dir.mkdir(exist_ok=True)
        self._init_artifacts()

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_artifacts(self) -> None:
        """Create any missing artifact files with minimal placeholder content.

        Called once in ``__init__``.  Existing files are never overwritten so
        Ada can resume a previous session without losing progress.
        """
        defaults: dict[str, str] = {
            "profile.md":      "# Project Profile\n\n_(to be filled by Ada in Phase 0)_\n",
            "mental_model.md": "# Mental Model\n\n_(to be filled by Ada in Phase 1)_\n",
            "baseline.json":   "{}\n",
            "backlog.md":      (
                "# Improvement Backlog\n\n"
                "| ID | Cat | Desc | Prio | Status |\n"
                "|----|-----|------|------|--------|\n"
            ),
            "journal.md":      "# Iteration Journal\n",
            "metrics.csv":     "timestamp,phase,metric,value\n",
            "questions.md":    "# Open Questions\n",
        }
        for name, content in defaults.items():
            p = self.ada_dir / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")

    # ── path sandbox ─────────────────────────────────────────────────────────

    def resolve(self, rel_or_abs: str) -> Path:
        """Resolve *rel_or_abs* to an absolute path inside ``target_dir``.

        Relative paths are resolved relative to ``target_dir``.  Absolute
        paths are accepted only if they fall inside ``target_dir``.

        Parameters
        ----------
        rel_or_abs:
            File path string as provided by the LLM tool call.

        Returns
        -------
        Path
            Canonicalised absolute path guaranteed to be inside ``target_dir``.

        Raises
        ------
        PermissionError
            If the resolved path escapes the target directory.
        """
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (self.target_dir / p).resolve()
        else:
            p = p.resolve()
        try:
            p.relative_to(self.target_dir)
        except ValueError as exc:
            raise PermissionError(
                f"Path escapes target directory: {p}"
            ) from exc
        return p

    # ── artifact helpers ─────────────────────────────────────────────────────

    def append_journal(self, text: str) -> None:
        """Append a timestamped Markdown section to ``journal.md``."""
        ts = datetime.now().isoformat(timespec="seconds")
        with (self.ada_dir / "journal.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {ts}\n{text}\n")

    def append_metric(self, phase: str, metric: str, value: str) -> None:
        """Append a single CSV row to ``metrics.csv``."""
        ts = datetime.now().isoformat(timespec="seconds")
        with (self.ada_dir / "metrics.csv").open("a", encoding="utf-8") as f:
            f.write(f"{ts},{phase},{metric},{value}\n")

    def write_baseline(self, data: dict) -> None:
        """Overwrite ``baseline.json`` with *data* (pretty-printed)."""
        (self.ada_dir / "baseline.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def read_artifact(self, name: str) -> str:
        """Return the full text of an artifact file."""
        return (self.ada_dir / name).read_text(encoding="utf-8")

    def write_artifact(self, name: str, content: str) -> None:
        """Overwrite an artifact file with *content*."""
        (self.ada_dir / name).write_text(content, encoding="utf-8")
