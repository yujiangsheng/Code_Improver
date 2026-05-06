"""Tool layer for Ada — all side effects on the target project go through here.

Each tool returns a dict (success path) or raises; the agent loop converts
exceptions to error strings for the LLM. All file paths are sandboxed inside
the workspace target_dir via Workspace.resolve().
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from .git_ops import Git, GitError
from .workspace import Workspace


class Tools:
    def __init__(self, ws: Workspace, cmd_timeout: int = 120,
                 planner: Callable[[str], str] | None = None):
        self.ws = ws
        self.cmd_timeout = cmd_timeout
        self.git = Git(ws.target_dir)
        self._planner = planner  # callable(prompt) -> str, may be None

    # ---- file ops ----------------------------------------------------------
    def list_dir(self, path: str = ".", max_entries: int = 500) -> dict[str, Any]:
        p = self.ws.resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(str(p))
        entries = []
        for i, child in enumerate(sorted(p.iterdir())):
            if i >= max_entries:
                entries.append("... (truncated)")
                break
            tag = "/" if child.is_dir() else ""
            entries.append(child.name + tag)
        return {"path": str(p.relative_to(self.ws.target_dir) or "."), "entries": entries}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, Any]:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        # cap size to avoid blowing context
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        if end - start + 1 > 800:
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
        p = self.ws.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p.relative_to(self.ws.target_dir)), "bytes": len(content)}

    def edit_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ValueError("old string not found")
        if count > 1:
            raise ValueError(f"old string is ambiguous (found {count} times)")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"path": str(p.relative_to(self.ws.target_dir)), "replaced": 1}

    def grep(self, pattern: str, path: str = ".", max_matches: int = 200) -> dict[str, Any]:
        import re
        regex = re.compile(pattern)
        root = self.ws.resolve(path)
        results = []
        for f in self._iter_text_files(root):
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if regex.search(line):
                        results.append(f"{f.relative_to(self.ws.target_dir)}:{i}: {line}")
                        if len(results) >= max_matches:
                            return {"matches": results, "truncated": True}
            except Exception:
                continue
        return {"matches": results, "truncated": False}

    def _iter_text_files(self, root: Path):
        skip = {".git", ".ada", "node_modules", ".venv", "venv", "__pycache__",
                "dist", "build", ".mypy_cache", ".pytest_cache"}
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.stat().st_size <= 1_000_000:  # 1 MB cap
                    yield p

    # ---- shell -------------------------------------------------------------
    def run_command(self, command: str, cwd: str | None = None,
                    timeout: int | None = None) -> dict[str, Any]:
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
        except subprocess.TimeoutExpired as e:
            return {
                "exit_code": -1,
                "stdout": (e.stdout or "")[-4000:],
                "stderr": f"[timeout after {e.timeout}s]\n" + (e.stderr or "")[-4000:],
            }
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-6000:],
            "stderr": proc.stderr[-4000:],
        }

    # ---- artifact convenience ---------------------------------------------
    def update_artifact(self, name: str, content: str) -> dict[str, Any]:
        if name not in Workspace.ARTIFACTS:
            raise ValueError(f"unknown artifact {name!r}; allowed: {Workspace.ARTIFACTS}")
        self.ws.write_artifact(name, content)
        return {"artifact": name, "bytes": len(content)}

    def append_journal(self, text: str) -> dict[str, Any]:
        self.ws.append_journal(text)
        return {"ok": True}

    def append_metric(self, phase: str, metric: str, value: str) -> dict[str, Any]:
        self.ws.append_metric(phase, metric, value)
        return {"ok": True}

    def read_artifact(self, name: str) -> dict[str, Any]:
        if name not in Workspace.ARTIFACTS:
            raise ValueError(f"unknown artifact {name!r}")
        return {"artifact": name, "content": self.ws.read_artifact(name)}

    # ---- git ---------------------------------------------------------------
    def _require_repo(self) -> None:
        if not self.git.is_repo():
            raise GitError("target directory is not a git repository")

    def git_status(self) -> dict[str, Any]:
        self._require_repo()
        return {"branch": self.git.current_branch(), "status": self.git.status()}

    def git_diff(self, paths: list[str] | None = None,
                 staged: bool = False) -> dict[str, Any]:
        self._require_repo()
        return {"diff": self.git.diff(paths=paths, staged=staged)}

    def git_create_branch(self, name: str) -> dict[str, Any]:
        self._require_repo()
        return {"branch": self.git.create_branch(name)}

    def git_commit(self, message: str) -> dict[str, Any]:
        self._require_repo()
        return self.git.commit_all(message)

    def git_revert(self, ref: str = "HEAD") -> dict[str, Any]:
        self._require_repo()
        return {"output": self.git.revert_to(ref)}

    # ---- multi-model: escalate to planner ----------------------------------
    def consult_planner(self, question: str, context: str = "") -> dict[str, Any]:
        if self._planner is None:
            return {"error": "no planner model configured"}
        prompt = (
            "You are Ada's senior planning advisor. Answer the worker's question "
            "with concrete, prioritised, actionable guidance. Be brief.\n\n"
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
