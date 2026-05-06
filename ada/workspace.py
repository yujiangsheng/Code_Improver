"""Workspace manager — handles the .ada/ directory under the target project.

Responsible for creating/maintaining Ada's persistent artifacts:
profile.md, mental_model.md, baseline.json, backlog.md, journal.md,
metrics.csv, questions.md.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


class Workspace:
    ARTIFACTS = [
        "profile.md",
        "mental_model.md",
        "baseline.json",
        "backlog.md",
        "journal.md",
        "metrics.csv",
        "questions.md",
    ]

    def __init__(self, target_dir: str | Path):
        self.target_dir = Path(target_dir).expanduser().resolve()
        if not self.target_dir.is_dir():
            raise NotADirectoryError(f"Target directory not found: {self.target_dir}")
        self.ada_dir = self.target_dir / ".ada"
        self.ada_dir.mkdir(exist_ok=True)
        self._init_artifacts()

    def _init_artifacts(self) -> None:
        defaults = {
            "profile.md": "# Project Profile\n\n_(to be filled by Ada in Phase 0)_\n",
            "mental_model.md": "# Mental Model\n\n_(to be filled by Ada in Phase 1)_\n",
            "baseline.json": "{}\n",
            "backlog.md": "# Improvement Backlog\n\n| ID | Cat | Desc | Prio | Status |\n|----|-----|------|------|--------|\n",
            "journal.md": "# Iteration Journal\n",
            "metrics.csv": "timestamp,phase,metric,value\n",
            "questions.md": "# Open Questions\n",
        }
        for name, content in defaults.items():
            p = self.ada_dir / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")

    # ---- safe path resolution ---------------------------------------------
    def resolve(self, rel_or_abs: str) -> Path:
        """Resolve a path; reject anything outside target_dir."""
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (self.target_dir / p).resolve()
        else:
            p = p.resolve()
        try:
            p.relative_to(self.target_dir)
        except ValueError as e:
            raise PermissionError(
                f"Path escapes target directory: {p}"
            ) from e
        return p

    # ---- artifact helpers --------------------------------------------------
    def append_journal(self, text: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with (self.ada_dir / "journal.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {ts}\n{text}\n")

    def append_metric(self, phase: str, metric: str, value: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with (self.ada_dir / "metrics.csv").open("a", encoding="utf-8") as f:
            f.write(f"{ts},{phase},{metric},{value}\n")

    def write_baseline(self, data: dict) -> None:
        (self.ada_dir / "baseline.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def read_artifact(self, name: str) -> str:
        return (self.ada_dir / name).read_text(encoding="utf-8")

    def write_artifact(self, name: str, content: str) -> None:
        (self.ada_dir / name).write_text(content, encoding="utf-8")
