"""Filesystem checkpoints: snapshot text files, rollback on failure.

Lightweight alternative to git stash for situations where:

* the project isn't a git repo,
* the agent wants to try a risky multi-file rewrite and revert atomically,
* or you just want a quick "save point" before a known-fragile step.

Snapshots live under ``.ada/checkpoints/<id>/`` and are pure file copies
keyed by relative path.  Each checkpoint also writes a ``manifest.json``
with the list of captured paths and a creation timestamp.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Hidden dirs we never snapshot even when explicitly requested — these
# bloat the on-disk store and rarely contain anything the agent should roll back.
_SKIP_DIRS = {".git", ".ada", "__pycache__", ".venv", "node_modules", ".tox"}


@dataclass
class Checkpoint:
    id: str
    root: Path
    files: list[str]
    created: float


class CheckpointStore:
    """Manage all snapshots for a single workspace root."""

    def __init__(self, ws_root: Path) -> None:
        self.ws_root = Path(ws_root)
        self.base = self.ws_root / ".ada" / "checkpoints"

    def create(
        self, paths: Iterable[str] | None = None, label: str = ""
    ) -> Checkpoint:
        """Snapshot *paths* (or every text file under root) to a new checkpoint.

        Returns the freshly-minted :class:`Checkpoint`.  Binary or oversized
        files (>1MB) are silently skipped — the goal is rollback of source,
        not full backup.
        """
        cp_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
        cp_dir = self.base / cp_id
        cp_dir.mkdir(parents=True, exist_ok=True)
        captured: list[str] = []
        for rel in self._iter_paths(paths):
            src = self.ws_root / rel
            if not src.is_file() or src.stat().st_size > 1_000_000:
                continue
            dst = cp_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                captured.append(rel)
            except OSError:
                continue
        manifest = {
            "id": cp_id,
            "label": label,
            "files": captured,
            "created": time.time(),
        }
        (cp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return Checkpoint(id=cp_id, root=cp_dir, files=captured, created=manifest["created"])

    def restore(self, cp_id: str) -> dict:
        """Restore every captured file from checkpoint *cp_id*.

        Files added since the checkpoint are NOT removed (we only roll back
        what we previously knew about).  Returns the restored count.
        """
        cp_dir = self.base / cp_id
        manifest_path = cp_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no checkpoint {cp_id!r}")
        manifest = json.loads(manifest_path.read_text())
        restored: list[str] = []
        for rel in manifest["files"]:
            src = cp_dir / rel
            dst = self.ws_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                restored.append(rel)
            except OSError:
                continue
        return {"id": cp_id, "restored": len(restored), "files": restored}

    def list(self) -> list[dict]:
        """Return manifests for every saved checkpoint, newest first."""
        if not self.base.is_dir():
            return []
        out: list[dict] = []
        for cp_dir in sorted(self.base.iterdir(), reverse=True):
            mf = cp_dir / "manifest.json"
            if mf.is_file():
                try:
                    out.append(json.loads(mf.read_text()))
                except json.JSONDecodeError:
                    continue
        return out

    def _iter_paths(self, paths: Iterable[str] | None) -> Iterable[str]:
        """Yield relative paths to capture (explicit list or full walk)."""
        if paths is not None:
            for p in paths:
                yield p
            return
        for dirpath, dirnames, filenames in os.walk(self.ws_root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                full = Path(dirpath) / fn
                yield str(full.relative_to(self.ws_root))
