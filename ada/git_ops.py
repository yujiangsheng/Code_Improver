"""Git operations for Ada — keep the human in control via branches & commits.

Design philosophy
-----------------
* **No gitpython dependency** — plain ``subprocess`` calls so this works
  anywhere git is installed without extra packages.
* **Ada always works on a dedicated branch** (``ada/<YYYYMMDD-HHmmSS>``).
  Your ``main`` / ``master`` branch is never touched.
* **One commit per verified Backlog item** — if a change later turns out to be
  wrong, ``git_revert`` rolls back to the last known-good state.
* **Diff size is capped** at 12 KB to keep LLM context manageable.

Error handling
--------------
All git failures raise ``GitError`` (a plain ``RuntimeError`` subclass).
The agent loop catches this and returns the message as a tool error string.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

# Maximum number of characters returned by diff() before truncation.
_DIFF_CAP = 12_000


class GitError(RuntimeError):
    """Raised when a git command exits with a non-zero status."""


def _run(
    cmd: list[str], cwd: Path, timeout: int = 30
) -> tuple[int, str, str]:
    """Run *cmd* in *cwd*, returning ``(returncode, stdout, stderr)``.

    Does **not** raise on non-zero exit — callers decide what to do.
    """
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout, proc.stderr


class Git:
    """Thin wrapper around the ``git`` CLI for a single repository.

    Parameters
    ----------
    repo_dir:
        Root of the git repository (the directory that contains ``.git``).
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = Path(repo_dir).resolve()

    # ── low-level helpers ────────────────────────────────────────────────────

    def is_repo(self) -> bool:
        """Return ``True`` if *repo_dir* is inside a git work-tree."""
        if not (self.repo_dir / ".git").exists():
            return False  # fast-path: no .git directory
        rc, _, _ = _run(
            ["git", "rev-parse", "--is-inside-work-tree"], self.repo_dir
        )
        return rc == 0

    def _git(self, *args: str, timeout: int = 30) -> str:
        """Run a git subcommand, raising ``GitError`` on failure.

        Returns stdout as a string; stderr is only surfaced on failure.
        """
        rc, out, err = _run(["git", *args], self.repo_dir, timeout=timeout)
        if rc != 0:
            cmd_str = " ".join(map(shlex.quote, args))
            raise GitError(
                f"git {cmd_str} failed: {err.strip() or out.strip()}"
            )
        return out

    # ── read operations ──────────────────────────────────────────────────────

    def current_branch(self) -> str:
        """Return the name of the currently checked-out branch."""
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def status(self) -> str:
        """Return the short-format status including branch info."""
        return self._git("status", "--short", "--branch")

    def changed_files(self, ref: str = "HEAD") -> list[str]:
        """Return paths modified vs *ref* (staged + unstaged + untracked).

        Combines ``git diff --name-only`` against *ref* with
        ``git ls-files --others --exclude-standard`` so newly-created
        files (which never appear in ``diff``) are still reported.
        """
        out: list[str] = []
        try:
            out.extend(
                p for p in self._git("diff", "--name-only", ref).splitlines() if p
            )
        except GitError:
            pass
        try:
            out.extend(
                p for p in self._git(
                    "ls-files", "--others", "--exclude-standard"
                ).splitlines() if p
            )
        except GitError:
            pass
        # Dedup, preserve order.
        seen: set[str] = set()
        return [p for p in out if not (p in seen or seen.add(p))]

    def diff(
        self, paths: list[str] | None = None, staged: bool = False
    ) -> str:
        """Return a combined stat + patch diff, capped at ``_DIFF_CAP`` chars.

        Parameters
        ----------
        paths:
            Limit the diff to specific files/directories.  ``None`` → all.
        staged:
            If ``True``, show the staged (index) diff instead of working tree.
        """
        args = ["diff", "--stat", "--patch"]
        if staged:
            args.append("--staged")
        if paths:
            args += ["--", *paths]
        out = self._git(*args, timeout=60)
        if len(out) >= _DIFF_CAP:
            return out[:_DIFF_CAP] + "\n... (diff truncated)"
        return out

    # ── write operations ─────────────────────────────────────────────────────

    def create_branch(self, name: str, checkout: bool = True) -> str:
        """Create a new branch, optionally checking it out immediately.

        Parameters
        ----------
        name:
            Branch name, e.g. ``ada/20260506-153000``.
        checkout:
            If ``True`` (default), switch to the new branch after creation.

        Returns
        -------
        str
            The branch name (same as *name*).
        """
        if checkout:
            self._git("checkout", "-b", name)
        else:
            self._git("branch", name)
        return name

    def commit_all(self, message: str) -> dict[str, Any]:
        """Stage all changes and commit them.

        Returns a dict describing the outcome:

        * ``{"committed": False, "reason": "no staged changes"}`` — nothing to commit.
        * ``{"committed": False, "reason": "<state> in progress"}`` — repo is mid-merge/rebase/cherry-pick.
        * ``{"committed": True, "sha": "<12-char SHA>", "message": "..."}`` — success.

        Raises
        ------
        GitError
            If ``git commit`` itself fails (e.g. no user.email configured).
        """
        # Refuse to commit while another operation is in flight; otherwise
        # the resulting commit silently completes a merge / rebase step the
        # user did not authorise.
        for marker, label in (
            ("MERGE_HEAD", "merge"),
            ("REBASE_HEAD", "rebase"),
            ("CHERRY_PICK_HEAD", "cherry-pick"),
            ("REVERT_HEAD", "revert"),
        ):
            if (self.repo_dir / ".git" / marker).exists():
                return {"committed": False, "reason": f"{label} in progress"}

        self._git("add", "-A")  # stage everything including new files

        # Check whether there is actually anything staged (avoids empty commits).
        rc, _, _ = _run(["git", "diff", "--cached", "--quiet"], self.repo_dir)
        if rc == 0:
            return {"committed": False, "reason": "no staged changes"}

        # Commit; message is passed as a single argument to support multi-line.
        rc, out, err = _run(
            ["git", "commit", "-m", message], self.repo_dir, timeout=30
        )
        if rc != 0:
            raise GitError(f"git commit failed: {err.strip() or out.strip()}")

        sha = self._git("rev-parse", "HEAD").strip()
        return {"committed": True, "sha": sha[:12], "message": message}

    def revert_to(self, ref: str = "HEAD") -> str:
        """Hard-reset the working tree to *ref*.

        .. warning::
            This is **destructive** — all uncommitted changes are discarded.
            Ada only calls this when a change fails verification and needs to
            be rolled back to the last good commit.

        Parameters
        ----------
        ref:
            Git ref to reset to.  Defaults to ``HEAD`` (discard working-tree
            changes while staying on the current commit).

        Returns
        -------
        str
            stdout from ``git reset --hard``.
        """
        return self._git("reset", "--hard", ref)
