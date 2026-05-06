"""Tool layer for Ada \u2014 all side effects on the target project go through here.

Tool design principles
----------------------
* **Sandboxed** \u2014 every file path is validated via ``Workspace.resolve()`` so
  the agent can never read or write outside the target directory.
* **Return dicts, not strings** \u2014 the agent loop serialises results to JSON and
  appends them to the conversation; structured data is easier for the LLM to
  parse than free text.
* **Raise on hard errors** \u2014 the agent loop catches all exceptions and converts
  them to ``{"error": "..."}`` JSON so the LLM can see what went wrong and
  self-correct.
* **Truncate large output** \u2014 ``read_file`` caps at 800 lines, ``grep`` caps
  at 200 matches, ``run_command`` caps stdout/stderr.  This keeps the
  conversation context manageable.

Tool catalogue
--------------
File I/O:     list_dir, read_file, write_file, edit_file, grep
Shell:        run_command
Artifacts:    update_artifact, append_journal, append_metric, read_artifact
Git:          git_status, git_diff, git_create_branch, git_commit, git_revert
Multi-model:  consult_planner
Control:      ask_user, finish   (intercepted by agent loop, not dispatched here)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from .git_ops import Git, GitError
from .workspace import Workspace

# Directories to skip when walking the file tree.
_SKIP_DIRS = {
    ".git", ".ada", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
}
# Largest file (bytes) read_file / grep will process.
_MAX_FILE_SIZE = 1_000_000  # 1 MB


class Tools:
    """Implements every tool callable by the Ada agent.

    Parameters
    ----------
    ws:
        Workspace instance (owns the ``.ada/`` directory and path sandbox).
    cmd_timeout:
        Default timeout in seconds for ``run_command``.  Individual calls may
        override with the ``timeout`` argument.
    planner:
        Optional callable ``(prompt: str) -> str`` that queries the planner
        model.  If ``None``, ``consult_planner`` returns an error message.
    """

    def __init__(
        self,
        ws: Workspace,
        cmd_timeout: int = 120,
        planner: Callable[[str], str] | None = None,
    ) -> None:
        self.ws = ws
        self.cmd_timeout = cmd_timeout
        self.git = Git(ws.target_dir)
        self._planner = planner

    # \u2500\u2500 file operations \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def list_dir(self, path: str = ".", max_entries: int = 500) -> dict[str, Any]:
        """List directory contents (sorted, directories suffixed with ``/``)."""
        p = self.ws.resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(str(p))
        entries: list[str] = []
        for i, child in enumerate(sorted(p.iterdir())):
            if i >= max_entries:
                entries.append("... (truncated)")
                break
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return {
            "path": str(p.relative_to(self.ws.target_dir) or "."),
            "entries": entries,
        }

    def read_file(
        self, path: str, start_line: int = 1, end_line: int = 400
    ) -> dict[str, Any]:
        """Read a slice of a text file with 1-based line numbers.

        Lines are returned as ``"  NNN: <content>"`` so the LLM can cite
        exact positions.  Max window is 800 lines per call; paginate with
        ``start_line`` / ``end_line`` for larger files.
        """
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        if end - start + 1 > 800:       # enforce hard cap
            end = start + 799
        snippet = "\n".join(
            f"{i:>5}: {lines[i - 1]}" for i in range(start, end + 1)
        )
        return {
            "path": str(p.relative_to(self.ws.target_dir)),
            "total_lines": len(lines),
            "shown": [start, end],
            "content": snippet,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Create or overwrite a file.

        Parent directories are created as needed.  Prefer ``edit_file`` for
        surgical changes; use this only when creating a new file or when the
        content changes so substantially that a targeted edit would be fragile.
        """
        p = self.ws.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p.relative_to(self.ws.target_dir)), "bytes": len(content)}

    def edit_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Replace **exactly one** occurrence of *old* with *new* in a file.

        Fails with a descriptive error if *old* is not found or appears more
        than once (ambiguous).  Include enough surrounding context in *old* to
        make it unique.
        """
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ValueError("old string not found in file")
        if count > 1:
            raise ValueError(
                f"old string is ambiguous \u2014 found {count} times; add more context"
            )
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"path": str(p.relative_to(self.ws.target_dir)), "replaced": 1}

    def grep(
        self, pattern: str, path: str = ".", max_matches: int = 200
    ) -> dict[str, Any]:
        """Regex search across text files, returning ``file:line: content`` hits.

        Binary files and files larger than 1 MB are silently skipped.
        Directories in ``_SKIP_DIRS`` (node_modules, .git, etc.) are excluded.
        """
        import re
        regex = re.compile(pattern)
        root = self.ws.resolve(path)
        results: list[str] = []
        for f in self._iter_text_files(root):
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if regex.search(line):
                        rel = f.relative_to(self.ws.target_dir)
                        results.append(f"{rel}:{i}: {line}")
                        if len(results) >= max_matches:
                            return {"matches": results, "truncated": True}
            except Exception:
                continue  # skip unreadable files
        return {"matches": results, "truncated": False}

    def _iter_text_files(self, root: Path):
        """Yield all text files under *root*, skipping common noise directories."""
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune in-place so os.walk doesn't descend into skipped dirs
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.stat().st_size <= _MAX_FILE_SIZE:
                    yield p

    # \u2500\u2500 shell \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute *command* in a subprocess inside the target project.

        stdout and stderr are tail-truncated (6 KB / 4 KB) to fit in context.
        On timeout, ``exit_code`` is ``-1`` and stderr contains the timeout
        notice.

        Parameters
        ----------
        command:
            Shell command string (``shell=True``).  Use ``&&`` to chain steps.
        cwd:
            Working directory relative to the target project.  Defaults to the
            project root.
        timeout:
            Seconds before the subprocess is killed.  Defaults to
            ``self.cmd_timeout``.
        """
        work = self.ws.resolve(cwd) if cwd else self.ws.target_dir
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(work),
                capture_output=True,
                text=True,
                timeout=timeout or self.cmd_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": -1,
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": f"[timeout after {exc.timeout}s]\n" + (exc.stderr or "")[-4000:],
            }
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-6000:],   # keep the tail; that's where results live
            "stderr": proc.stderr[-4000:],
        }

    # \u2500\u2500 artifact helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def update_artifact(self, name: str, content: str) -> dict[str, Any]:
        """Overwrite one of Ada's ``.ada/`` workspace artifact files."""
        if name not in Workspace.ARTIFACTS:
            raise ValueError(
                f"unknown artifact {name!r}; allowed: {Workspace.ARTIFACTS}"
            )
        self.ws.write_artifact(name, content)
        return {"artifact": name, "bytes": len(content)}

    def append_journal(self, text: str) -> dict[str, Any]:
        """Append a timestamped entry to ``.ada/journal.md``."""
        self.ws.append_journal(text)
        return {"ok": True}

    def append_metric(self, phase: str, metric: str, value: str) -> dict[str, Any]:
        """Append a CSV row to ``.ada/metrics.csv``."""
        self.ws.append_metric(phase, metric, value)
        return {"ok": True}

    def read_artifact(self, name: str) -> dict[str, Any]:
        """Return the full content of a ``.ada/`` artifact file."""
        if name not in Workspace.ARTIFACTS:
            raise ValueError(f"unknown artifact {name!r}")
        return {"artifact": name, "content": self.ws.read_artifact(name)}

    # \u2500\u2500 git \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _require_repo(self) -> None:
        """Raise ``GitError`` if the target directory is not a git repository."""
        if not self.git.is_repo():
            raise GitError("target directory is not a git repository")

    def git_status(self) -> dict[str, Any]:
        """Return current branch name and short working-tree status."""
        self._require_repo()
        return {"branch": self.git.current_branch(), "status": self.git.status()}

    def git_diff(
        self, paths: list[str] | None = None, staged: bool = False
    ) -> dict[str, Any]:
        """Return a stat+patch diff (truncated at 12 KB)."""
        self._require_repo()
        return {"diff": self.git.diff(paths=paths, staged=staged)}

    def git_create_branch(self, name: str) -> dict[str, Any]:
        """Create and check out a new branch."""
        self._require_repo()
        return {"branch": self.git.create_branch(name)}

    def git_commit(self, message: str) -> dict[str, Any]:
        """Stage all changes and commit.  Returns commit SHA on success."""
        self._require_repo()
        return self.git.commit_all(message)

    def git_revert(self, ref: str = "HEAD") -> dict[str, Any]:
        """Hard-reset the working tree to *ref*.  **Destructive.**"""
        self._require_repo()
        return {"output": self.git.revert_to(ref)}

    # \u2500\u2500 multi-model escalation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def consult_planner(self, question: str, context: str = "") -> dict[str, Any]:
        """Escalate a hard question to the stronger planner model.

        The worker calls this when it needs architectural guidance, backlog
        prioritisation, or root-cause analysis for a tricky bug.  The planner
        receives a structured prompt and returns plain-text advice.

        Parameters
        ----------
        question:
            The specific question for the planner.
        context:
            Relevant excerpts (code, error messages, metrics) to give the
            planner enough context to answer well.  Keep it concise.
        """
        if self._planner is None:
            return {"error": "no planner model configured"}
        prompt = (
            "You are Ada's senior planning advisor.  Answer the worker\u2019s question "
            "with concrete, prioritised, actionable guidance.  Be brief.\n\n"
            f"=== Context ===\n{context.strip() or '(none)'}\n\n"
            f"=== Question ===\n{question.strip()}"
        )
        return {"advice": self._planner(prompt)}


# ---- OpenAI tool schema --------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory (relative to target project). Trailing '/' marks directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_entries": {"type": "integer", "default": 500},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file with line numbers. Use start_line/end_line to page through large files (max 800 lines per call).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer", "default": 400},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with new content. Use sparingly; prefer edit_file for surgical changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one occurrence of `old` with `new` in a file. Fails if `old` is missing or ambiguous. Include enough context in `old` to be unique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search across text files under a path. Returns 'file:line: content' matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_matches": {"type": "integer", "default": 200},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the target project. Use for: install deps, build, run tests, lint, benchmarks. stdout/stderr are tail-truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_artifact",
            "description": "Overwrite one of Ada's workspace artifacts in .ada/. Allowed names: profile.md, mental_model.md, baseline.json, backlog.md, journal.md, metrics.csv, questions.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_journal",
            "description": "Append a timestamped entry to .ada/journal.md.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_metric",
            "description": "Append one row to .ada/metrics.csv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "metric": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["phase", "metric", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": "Read one of Ada's workspace artifacts.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show current branch and short status of the target git repo.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show working-tree diff (or staged diff). Truncated at 12 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "staged": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Create and check out a new branch.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and commit with the given message. Use after each verified Backlog item to keep history clean & revertible.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_revert",
            "description": "Hard-reset the working tree to a ref (default HEAD). DESTRUCTIVE — use only to discard a failed unverified change.",
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "string", "default": "HEAD"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_planner",
            "description": "Escalate a hard planning/design question to the stronger planner model. Use for: prioritising the backlog, root-causing tricky bugs, architectural decisions. Pass concise relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Pause and ask the human for clarification, approval, or feedback. Use at Phase 7 reports, on uncertainty, or before risky actions.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the task complete. Provide a final summary string.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]
