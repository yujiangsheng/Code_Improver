"""Git helpers for Ada — keep the human in control via branches & commits."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


class GitError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout, proc.stderr


class Git:
    def __init__(self, repo_dir: Path):
        self.repo_dir = Path(repo_dir).resolve()

    # ---- low level ---------------------------------------------------------
    def is_repo(self) -> bool:
        if not (self.repo_dir / ".git").exists():
            return False
        rc, _, _ = _run(["git", "rev-parse", "--is-inside-work-tree"], self.repo_dir)
        return rc == 0

    def _git(self, *args: str, timeout: int = 30) -> str:
        rc, out, err = _run(["git", *args], self.repo_dir, timeout=timeout)
        if rc != 0:
            raise GitError(f"git {' '.join(map(shlex.quote, args))} failed: {err.strip() or out.strip()}")
        return out

    # ---- ops ---------------------------------------------------------------
    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def status(self) -> str:
        return self._git("status", "--short", "--branch")

    def diff(self, paths: list[str] | None = None, staged: bool = False) -> str:
        args = ["diff", "--stat", "--patch"]
        if staged:
            args.append("--staged")
        if paths:
            args.append("--")
            args.extend(paths)
        out = self._git(*args, timeout=60)
        # cap diff size
        return out if len(out) < 12000 else out[:12000] + "\n... (diff truncated)"

    def create_branch(self, name: str, checkout: bool = True) -> str:
        if checkout:
            self._git("checkout", "-b", name)
        else:
            self._git("branch", name)
        return name

    def commit_all(self, message: str) -> dict[str, Any]:
        # Stage everything then commit; allow-empty=False.
        self._git("add", "-A")
        rc, out, err = _run(
            ["git", "diff", "--cached", "--quiet"], self.repo_dir
        )
        if rc == 0:
            return {"committed": False, "reason": "no staged changes"}
        # use -m via args; commit message can contain newlines
        rc, out, err = _run(
            ["git", "commit", "-m", message], self.repo_dir, timeout=30
        )
        if rc != 0:
            raise GitError(f"git commit failed: {err.strip() or out.strip()}")
        sha = self._git("rev-parse", "HEAD").strip()
        return {"committed": True, "sha": sha[:12], "message": message}

    def revert_to(self, ref: str = "HEAD") -> str:
        # hard-reset working tree to ref (CAUTION).
        return self._git("reset", "--hard", ref)
