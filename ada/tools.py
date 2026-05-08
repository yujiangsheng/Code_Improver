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
File I/O:     list_dir, read_file, write_file, edit_file, ast_edit, patch_apply, grep
Semantic:     list_symbols, find_definition, find_references, ts_edit, read_symbol, repo_map  (tree-sitter)
Embedding:    semantic_search, reindex_embeddings, embed_stats  (vector index)
Verify:       run_tests, run_lint, detect_toolchain  (auto-detected toolchains)
Safety:       scan_secrets  (run_command screens dangerous patterns by default)
Shell:        run_command
Artifacts:    update_artifact, append_journal, append_metric, read_artifact
Git:          git_status, git_diff, git_create_branch, git_commit, git_revert
Multi-model:  consult_planner
MCP:          mcp_status  + dynamic mcp_<server>_<tool> per .ada/mcp.json
Control:      ask_user, finish   (intercepted by agent loop, not dispatched here)
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointStore
from .embed import EmbedIndex
from .git_ops import Git, GitError
from .impact import changed_test_targets, diff_stats
from .mcp import MCPClient
from .memstore import MemoryStore
from .notebook import edit_cell_source as _nb_edit, read_notebook as _nb_read
from .profile import profile_python as _profile_python
from .refactor import rename_symbol as _rename_symbol
from .safety import assess_command, scan_secrets
from .semantic import Semantic
from .tasks import TaskQueue
from .traceback_parser import in_workspace, parse as parse_traceback
from .verify import Verifier
from .web import fetch_url as _fetch_url
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
        # Multi-language semantic engine (tree-sitter).  Lazily usable: when
        # the optional ``tree_sitter_language_pack`` is missing every method
        # returns a structured ``{"error": "tree-sitter not available"}``.
        self._sem = Semantic(ws.target_dir)
        # Auto-detected toolchain runner (pytest / ruff / mypy / eslint / ...).
        self._verifier = Verifier(ws.target_dir, default_timeout=cmd_timeout)
        # Last shell command that was blocked by the safety guard, so the
        # model can opt-in by repeating it via run_command(force=True).
        self._last_blocked: str | None = None
        # Embedding-based semantic search.  Lazy: returns an "unavailable"
        # error from semantic_search when no embedder backend is configured.
        self._embed = EmbedIndex(ws.target_dir)
        # MCP (Model Context Protocol) client.  Looks for ``.ada/mcp.json``
        # at construction time; missing config is a no-op.
        self._mcp = MCPClient.from_config(ws.target_dir / ".ada" / "mcp.json")
        # Read-file cache to avoid re-paying tokens for the same file slice.
        # Keyed by (abs_path, mtime, size, start_line, end_line).  Reset on
        # any mutating tool that targets the same file (write/edit/ts_edit).
        self._read_cache: dict[tuple, dict[str, Any]] = {}
        self._read_cache_hits = 0
        # Filesystem checkpoint store for snapshot/rollback.
        self._checkpoints = CheckpointStore(ws.target_dir)
        # Persistent task queue (stored at .ada/tasks.json).
        self._tasks = TaskQueue(ws.target_dir / ".ada" / "tasks.json")
        # SQLite-backed cross-run K/V memory.
        self._memory = MemoryStore(ws.target_dir / ".ada" / "memory.sqlite")

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

        Cached by ``(path, mtime, size, start, end)``: identical re-reads
        return ``{"cached": True, ...}`` instead of re-loading the file,
        nudging the model to vary its window or move on.
        """
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        st = p.stat()
        key = (str(p), st.st_mtime, st.st_size, start_line, end_line)
        cached = self._read_cache.get(key)
        if cached is not None:
            self._read_cache_hits += 1
            return {**cached, "cached": True,
                    "hint": "you already read this exact slice; vary the window or move on"}
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        if end - start + 1 > 800:       # enforce hard cap
            end = start + 799
        snippet = "\n".join(
            f"{i:>5}: {lines[i - 1]}" for i in range(start, end + 1)
        )
        result = {
            "path": str(p.relative_to(self.ws.target_dir)),
            "total_lines": len(lines),
            "shown": [start, end],
            "content": snippet,
        }
        self._read_cache[key] = result
        return result

    def write_file(self, path: str, content: str, allow_secrets: bool = False) -> dict[str, Any]:
        """Create or overwrite a file.

        Parent directories are created as needed.  Prefer ``edit_file`` for
        surgical changes; use this only when creating a new file or when the
        content changes so substantially that a targeted edit would be fragile.

        When ``ADA_GUARD_SECRETS=1`` (default) any new content is scanned
        for credential patterns and the write is refused unless
        ``allow_secrets=True`` is explicitly passed.
        """
        guarded = self._guard_write(content, allow_secrets)
        if guarded is not None:
            return guarded
        p = self.ws.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p.relative_to(self.ws.target_dir)), "bytes": len(content)}

    def _guard_write(self, content: str, allow_secrets: bool) -> dict[str, Any] | None:
        """Block writes that look like they're checking in credentials.

        Returns an error dict to short-circuit the caller, or ``None`` when
        the write is safe to proceed.  No-op when ``ADA_GUARD_SECRETS=0``.
        """
        if allow_secrets:
            return None
        if os.getenv("ADA_GUARD_SECRETS", "1").lower() in ("0", "false", "no", "off"):
            return None
        hits = scan_secrets(content)
        if not hits:
            return None
        return {
            "error": "secret_guard: refusing to write content that looks like credentials",
            "hits": [
                {"kind": h.kind, "line": h.line, "snippet": h.snippet}
                for h in hits[:10]
            ],
            "hint": "remove the secret, or pass allow_secrets=true if this is a fixture",
        }

    def edit_file(self, path: str, old: str, new: str, allow_secrets: bool = False) -> dict[str, Any]:
        """Replace **exactly one** occurrence of *old* with *new* in a file.

        Fails with a descriptive error if *old* is not found or appears more
        than once (ambiguous).  Include enough surrounding context in *old* to
        make it unique.  Honours the same secret-guard as ``write_file``.
        """
        guarded = self._guard_write(new, allow_secrets)
        if guarded is not None:
            return guarded
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

    def ast_edit(
        self,
        path: str,
        operation: str,
        target: str = "",
        code: str = "",
    ) -> dict[str, Any]:
        """Structural Python edit by symbol name (zero-dependency).

        Uses the stdlib ``ast`` module to locate top-level functions, classes
        and methods (dotted ``Class.method`` paths supported, one level deep)
        and rewrites the source by line range so formatting outside the
        target span is preserved exactly.

        Parameters
        ----------
        path:
            Python file relative to the target project.  Must end in ``.py``.
        operation:
            One of:
              * ``"show"``           — return the source span of *target*
              * ``"replace"``        — replace the def/class with *code*
              * ``"delete"``         — remove the def/class
              * ``"insert_before"``  — insert *code* immediately before *target*
              * ``"insert_after"``   — insert *code* immediately after *target*
              * ``"append"``         — append *code* at end of file (no target)
        target:
            Symbol name.  Either ``"foo"`` for a top-level function/class or
            ``"MyClass.bar"`` for a method/nested class.  Required for every
            operation except ``"append"``.
        code:
            New source code (multi-line OK).  Required for ``replace``,
            ``insert_*`` and ``append``.  Must parse as valid Python and is
            re-indented to match the target's column.

        Returns
        -------
        dict
            ``{"path": ..., "operation": ..., "target": ..., "lines": [start, end]}``
            On ``"show"``, also includes ``"source"``.
        """
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        if p.suffix != ".py":
            raise ValueError("ast_edit only supports .py files")

        text = p.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        # ── append: no target lookup needed ─────────────────────────────
        if operation == "append":
            if not code.strip():
                raise ValueError("code is required for append")
            ast.parse(code)  # validate
            new_text = text
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
            if not new_text.endswith("\n\n"):
                new_text += "\n"
            new_text += code.rstrip() + "\n"
            p.write_text(new_text, encoding="utf-8")
            return {
                "path": str(p.relative_to(self.ws.target_dir)),
                "operation": operation,
                "target": "",
                "lines": [len(lines) + 1, len(new_text.splitlines())],
            }

        if not target:
            raise ValueError(f"operation {operation!r} requires a target")

        # ── locate the target node ──────────────────────────────────────
        try:
            tree = ast.parse(text, filename=str(p))
        except SyntaxError as exc:
            raise ValueError(f"file has syntax error: {exc}") from exc

        node = self._find_symbol(tree, target)
        if node is None:
            raise ValueError(f"symbol {target!r} not found in {path}")

        # Span includes any decorators above the def.
        start_line = node.lineno
        if getattr(node, "decorator_list", None):
            start_line = min(d.lineno for d in node.decorator_list)
        end_line = getattr(node, "end_lineno", node.lineno)
        col = node.col_offset
        indent = " " * col

        if operation == "show":
            source = "".join(lines[start_line - 1 : end_line])
            return {
                "path": str(p.relative_to(self.ws.target_dir)),
                "operation": operation,
                "target": target,
                "lines": [start_line, end_line],
                "source": source,
            }

        # All remaining operations mutate the file.
        before = lines[: start_line - 1]
        target_block = lines[start_line - 1 : end_line]
        after = lines[end_line:]

        if operation == "delete":
            new_lines = before + after
            new_text = "".join(new_lines)
            p.write_text(new_text, encoding="utf-8")
            return {
                "path": str(p.relative_to(self.ws.target_dir)),
                "operation": operation,
                "target": target,
                "lines": [start_line, end_line],
            }

        if not code.strip():
            raise ValueError(f"code is required for operation {operation!r}")
        ast.parse(code)  # validate replacement is parseable Python
        new_block = self._reindent(code, indent) + "\n"

        if operation == "replace":
            new_lines = before + [new_block] + after
        elif operation == "insert_before":
            # Place a blank line between insertion and target if there isn't
            # already vertical whitespace separating them.
            sep = "" if before and before[-1].strip() == "" else "\n"
            new_lines = before + [new_block, sep] + target_block + after
        elif operation == "insert_after":
            sep = "" if after and after[0].strip() == "" else "\n"
            new_lines = before + target_block + [sep, new_block] + after
        else:
            raise ValueError(f"unknown ast_edit operation: {operation!r}")

        new_text = "".join(new_lines)
        # Final validation: full file must still parse.
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            raise ValueError(
                f"edit would produce invalid Python ({exc}); aborted"
            ) from exc
        p.write_text(new_text, encoding="utf-8")

        new_total = len(new_text.splitlines())
        return {
            "path": str(p.relative_to(self.ws.target_dir)),
            "operation": operation,
            "target": target,
            "lines": [start_line, end_line],
            "new_total_lines": new_total,
        }

    @staticmethod
    def _find_symbol(tree: ast.Module, target: str) -> ast.AST | None:
        """Resolve a dotted ``target`` name to an ``ast`` node.

        Supports one level of nesting (``Class.method`` or ``Class.Inner``).
        Matches ``FunctionDef``, ``AsyncFunctionDef`` and ``ClassDef``.
        """
        parts = target.split(".")

        def _children(node: ast.AST) -> list[ast.AST]:
            return list(getattr(node, "body", []))

        scope: list[ast.AST] = list(tree.body)
        found: ast.AST | None = None
        for part in parts:
            found = None
            for child in scope:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ) and child.name == part:
                    found = child
                    break
            if found is None:
                return None
            scope = _children(found)
        return found

    @staticmethod
    def _reindent(code: str, indent: str) -> str:
        """Re-indent *code* so its outermost block sits at column ``len(indent)``.

        Strips common leading whitespace first (so callers can pass naturally
        indented snippets) then prefixes every non-empty line with *indent*.
        Trailing whitespace on each line is preserved as-is.
        """
        import textwrap
        dedented = textwrap.dedent(code).rstrip("\n")
        if not indent:
            return dedented
        return "\n".join(
            (indent + ln) if ln.strip() else ln
            for ln in dedented.splitlines()
        )

    # ── unified diff patching ─────────────────────────────────────────────

    def patch_apply(
        self,
        patch: str,
        check_only: bool = False,
    ) -> dict[str, Any]:
        """Apply a unified-diff patch across one or more files atomically.

        Accepts standard ``diff -u`` / ``git diff`` output: each file block
        starts with ``--- <old>`` / ``+++ <new>`` headers (any ``a/`` / ``b/``
        prefixes are stripped) and contains one or more ``@@ ... @@`` hunks.
        Hunks are applied **by content**: the leading context + ``-`` lines
        must match exactly somewhere in the file (the ``@@`` line numbers are
        used only as a hint, then we search ±200 lines to tolerate drift).

        Special cases:
        * ``--- /dev/null``                → file is **created** (use ``+`` lines only)
        * ``+++ /dev/null``                → file is **deleted**
        * Hunks with only ``+`` and `` `` lines  → pure insertion
        * Hunks with only ``-`` and `` `` lines  → pure deletion

        All edits are computed in memory first; if any hunk fails to apply
        nothing is written.  Use ``check_only=True`` for a dry run.

        Parameters
        ----------
        patch:
            Full unified-diff text.  May contain multiple file sections.
        check_only:
            If True, validate but do not write.

        Returns
        -------
        dict with ``files`` (per-file outcome) and ``applied`` / ``failed`` counts.
        """
        if not patch.strip():
            raise ValueError("empty patch")

        files = self._split_patch(patch)
        if not files:
            raise ValueError("no file sections found in patch")

        plan: list[tuple[Path, str | None, str]] = []  # (path, new_text or None for delete, action)
        per_file: list[dict[str, Any]] = []
        for old_path, new_path, hunks in files:
            action, target_rel, new_text = self._apply_file_patch(
                old_path, new_path, hunks
            )
            target_abs = self.ws.resolve(target_rel)
            plan.append((target_abs, new_text, action))
            per_file.append({
                "path": str(target_abs.relative_to(self.ws.target_dir)),
                "action": action,
                "hunks": len(hunks),
            })

        if check_only:
            return {"files": per_file, "applied": 0, "checked": len(per_file)}

        # Atomic write phase: only reached if every file applied cleanly above.
        for path, new_text, action in plan:
            if action == "delete":
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_text or "", encoding="utf-8")
        return {"files": per_file, "applied": len(per_file), "failed": 0}

    @staticmethod
    def _split_patch(patch: str) -> list[tuple[str, str, list[list[str]]]]:
        """Parse unified-diff text into ``[(old_path, new_path, hunks)]``.

        ``hunks`` is a list of hunks; each hunk is a list of raw lines
        (including the leading ``@@`` header) preserved verbatim.
        """
        lines = patch.splitlines()
        files: list[tuple[str, str, list[list[str]]]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("--- "):
                old_path = line[4:].split("\t", 1)[0].strip()
                if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                    raise ValueError(f"malformed patch near line {i + 1}: missing +++")
                new_path = lines[i + 1][4:].split("\t", 1)[0].strip()
                i += 2
                hunks: list[list[str]] = []
                while i < len(lines) and lines[i].startswith("@@"):
                    hunk = [lines[i]]
                    i += 1
                    while i < len(lines) and not (
                        lines[i].startswith("@@")
                        or lines[i].startswith("--- ")
                        or lines[i].startswith("diff ")
                    ):
                        hunk.append(lines[i])
                        i += 1
                    hunks.append(hunk)
                if not hunks:
                    raise ValueError(f"file section {old_path!r} has no hunks")
                files.append((old_path, new_path, hunks))
            else:
                i += 1
        return files

    @staticmethod
    def _strip_prefix(p: str) -> str:
        """Strip git-style ``a/`` / ``b/`` prefixes; pass through ``/dev/null``."""
        if p == "/dev/null":
            return p
        for pre in ("a/", "b/"):
            if p.startswith(pre):
                return p[len(pre):]
        return p

    def _apply_file_patch(
        self,
        old_path: str,
        new_path: str,
        hunks: list[list[str]],
    ) -> tuple[str, str, str | None]:
        """Apply *hunks* to one file; return ``(action, target_rel, new_text|None)``.

        ``action`` is ``"create"`` / ``"modify"`` / ``"delete"``.
        ``new_text`` is None when the file is deleted.
        Raises ``ValueError`` if any hunk does not match.
        """
        old_p = self._strip_prefix(old_path)
        new_p = self._strip_prefix(new_path)

        # Deletion: +++ /dev/null
        if new_p == "/dev/null":
            target_rel = old_p
            return "delete", target_rel, None

        # Creation: --- /dev/null
        if old_p == "/dev/null":
            target_rel = new_p
            content_lines: list[str] = []
            for hunk in hunks:
                for ln in hunk[1:]:  # skip @@ header
                    if ln.startswith("+"):
                        content_lines.append(ln[1:])
                    elif ln.startswith(" "):
                        content_lines.append(ln[1:])
                    elif ln.startswith("-"):
                        raise ValueError(
                            f"creation hunk for {new_p!r} has '-' lines"
                        )
                    # ignore '\ No newline at end of file' markers
            return "create", target_rel, "\n".join(content_lines) + (
                "\n" if content_lines else ""
            )

        # Modify: locate, splice, repeat for each hunk.
        target_rel = new_p
        target_abs = self.ws.resolve(old_p)
        if not target_abs.is_file():
            raise FileNotFoundError(f"patch target not found: {old_p}")
        original = target_abs.read_text(encoding="utf-8")
        had_trailing_nl = original.endswith("\n")
        cur_lines = original.splitlines()

        for hunk in hunks:
            cur_lines = self._splice_hunk(cur_lines, hunk, old_p)

        new_text = "\n".join(cur_lines)
        if had_trailing_nl or new_text == "":
            new_text += "\n" if not new_text.endswith("\n") else ""
        return "modify", target_rel, new_text

    @staticmethod
    def _parse_hunk_header(header: str) -> tuple[int, int]:
        """Parse ``@@ -a,b +c,d @@`` → ``(old_start_1based, old_count)``.

        ``,b`` is optional (defaults to 1).  Used only as a search hint.
        """
        import re
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not m:
            raise ValueError(f"bad hunk header: {header!r}")
        return int(m.group(1)), int(m.group(2) or "1")

    def _splice_hunk(
        self, cur_lines: list[str], hunk: list[str], path_for_err: str
    ) -> list[str]:
        """Apply a single hunk to *cur_lines* and return the new line list.

        We extract the "expected old block" (context + ``-`` lines) and the
        "new block" (context + ``+`` lines), then locate the old block in
        ``cur_lines`` near the header's hint position (search ±200 lines).
        Exact match required; raises ``ValueError`` on miss.
        """
        old_start, _ = self._parse_hunk_header(hunk[0])
        old_block: list[str] = []
        new_block: list[str] = []
        for raw in hunk[1:]:
            if not raw:
                # An empty raw line in a hunk represents an empty context line.
                old_block.append("")
                new_block.append("")
                continue
            tag, body = raw[0], raw[1:]
            if tag == " ":
                old_block.append(body)
                new_block.append(body)
            elif tag == "-":
                old_block.append(body)
            elif tag == "+":
                new_block.append(body)
            elif tag == "\\":
                # "\ No newline at end of file" — ignore for splicing purposes.
                continue
            else:
                # Unknown tag: tolerate by treating it as context to be lenient.
                old_block.append(raw)
                new_block.append(raw)

        # Search for old_block in cur_lines near the hinted position.
        hint = max(0, old_start - 1)
        n = len(cur_lines)
        m = len(old_block)
        if m == 0:
            # Pure insertion at hint.
            idx = min(hint, n)
            return cur_lines[:idx] + new_block + cur_lines[idx:]

        # Try exact hint first, then expanding window ±200.
        candidates = [hint]
        for delta in range(1, 201):
            if hint - delta >= 0:
                candidates.append(hint - delta)
            if hint + delta + m <= n:
                candidates.append(hint + delta)
        for idx in candidates:
            if 0 <= idx <= n - m and cur_lines[idx:idx + m] == old_block:
                return cur_lines[:idx] + new_block + cur_lines[idx + m:]

        # Fallback: scan the whole file once in case drift exceeded the window.
        for idx in range(0, n - m + 1):
            if cur_lines[idx:idx + m] == old_block:
                return cur_lines[:idx] + new_block + cur_lines[idx + m:]

        raise ValueError(
            f"hunk did not apply to {path_for_err} "
            f"(near line {old_start}); file may have diverged from the patch base"
        )

    # ── semantic (tree-sitter) ──────────────────────────────────────────

    def _require_ts(self) -> dict[str, Any] | None:
        """Return an error dict when tree-sitter isn't installed; else None."""
        if not Semantic.available():
            return {
                "error": (
                    "tree-sitter not available — install with "
                    "`pip install tree-sitter-language-pack` to enable "
                    "list_symbols / find_definition / find_references / ts_edit"
                ),
            }
        return None

    def list_symbols(self, path: str) -> dict[str, Any]:
        """List every named definition in a source file (tree-sitter).

        Returns one entry per function/class/method/etc., with a dotted
        ``qualified_name`` for nested scopes.  Supported extensions:
        py, js, mjs, cjs, jsx, ts, tsx, go, rs, java, c, h, cpp, cc, cxx,
        hpp, hh.
        """
        err = self._require_ts()
        if err is not None:
            return err
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        syms = self._sem.list_symbols(p)
        return {
            "path": str(p.relative_to(self.ws.target_dir)),
            "language": p.suffix.lstrip("."),
            "symbols": [
                {
                    "name": s.name,
                    "qualified_name": s.qualified_name,
                    "kind": s.kind,
                    "lines": [s.start_line, s.end_line],
                }
                for s in syms
            ],
        }

    def find_definition(self, name: str, path: str = ".") -> dict[str, Any]:
        """Project-wide: every definition whose leaf name equals *name*.

        Walks the whole subtree under *path* (or the project root), parses
        every supported source file, and returns each definition's location.
        """
        err = self._require_ts()
        if err is not None:
            return err
        root = self.ws.resolve(path)
        results = self._sem.find_definition(name, root)
        return {"name": name, "matches": results}

    def find_references(
        self,
        name: str,
        path: str = ".",
        max_results: int = 500,
    ) -> dict[str, Any]:
        """Project-wide: identifier-position occurrences of *name* (tree-sitter).

        Beats ``grep`` because matches inside string literals, comments, and
        unrelated tokens are excluded automatically.  NOT scope-aware — local
        variables that shadow *name* are still reported (LSP would solve
        this; out of scope for this implementation).
        """
        err = self._require_ts()
        if err is not None:
            return err
        root = self.ws.resolve(path)
        hits = self._sem.find_references(name, root, max_results=max_results)
        return {
            "name": name,
            "matches": hits,
            "truncated": len(hits) >= max_results,
        }

    def ts_edit(
        self,
        path: str,
        operation: str,
        symbol: str = "",
        code: str = "",
    ) -> dict[str, Any]:
        """Multi-language structural edit by symbol name (tree-sitter).

        The cross-language counterpart to ``ast_edit`` — same operation set
        (``show`` / ``replace`` / ``delete`` / ``insert_before`` /
        ``insert_after``) but works on Python, JS/TS, Go, Rust, Java, C/C++.
        After writing, the file is re-parsed; if the edit introduces new
        ``ERROR``/``MISSING`` parse nodes, the change is rolled back.

        Use ``ast_edit`` for Python (cheaper, supports the ``append``
        operation); use ``ts_edit`` for everything else.
        """
        if operation == "append":
            raise ValueError(
                "ts_edit does not support 'append' (use write_file or ast_edit)"
            )
        err = self._require_ts()
        if err is not None:
            return err
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return self._sem.edit_symbol(p, symbol, operation, code)


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
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute *command* in a subprocess inside the target project.

        stdout and stderr are tail-truncated (6 KB / 4 KB) to fit in context.
        On timeout, ``exit_code`` is ``-1`` and stderr contains the timeout
        notice.

        Safety
        ------
        Before running, the command is screened by
        :py:func:`ada.safety.assess_command`.  If a dangerous pattern matches
        (e.g. ``rm -rf /``, ``git push --force``, ``curl … | sh``) execution
        is **refused** and a structured ``{"blocked": True, ...}`` dict is
        returned.  Set ``force=True`` to override after explicit human
        confirmation, or ``ADA_SAFETY=0`` to disable the guard globally.

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
        force:
            Bypass the safety guard for *this* call only.  Use sparingly.
        """
        if not force:
            risk = assess_command(command)
            if risk.is_dangerous:
                self._last_blocked = command
                return risk.to_error()
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

    # ── verification (auto-detected toolchains) ──────────────────────────

    def run_tests(
        self, only: list[str] | None = None, timeout: int | None = None
    ) -> dict[str, Any]:
        """Auto-detect and run the project's test suite(s).

        Detects ``pytest`` / ``unittest`` / ``go test`` / ``cargo test`` /
        ``npm test`` from marker files and runs every matching one.  Returns
        a structured result with per-tool exit codes and tail-truncated
        stdout/stderr so the model can react to failures.

        Pass ``only=["pytest"]`` to restrict the run to a subset.
        """
        return self._verifier.run_tests(only=only, timeout=timeout)

    def run_lint(
        self, only: list[str] | None = None, timeout: int | None = None
    ) -> dict[str, Any]:
        """Auto-detect and run lint / typecheck tools (ruff/mypy/eslint/tsc/…).

        Same shape as :py:meth:`run_tests`.  Use after edits to catch
        regressions before committing.
        """
        return self._verifier.run_lint(only=only, timeout=timeout)

    def detect_toolchain(self) -> dict[str, Any]:
        """List the test / lint / format toolchains detected in this repo."""
        return self._verifier.detect()

    # ── targeted reads (token-saving) ────────────────────────────────────

    def read_symbol(self, path: str, symbol: str) -> dict[str, Any]:
        """Return the source of one symbol (cheaper than ``read_file``).

        Use this once you know which function/class/method you care about —
        it costs ~20 lines of context vs. a full file dump.  Combines
        :py:meth:`list_symbols` (to locate the span) with a precise read.
        """
        if not Semantic.available():
            return {
                "error": (
                    "tree-sitter not available — install tree-sitter-language-pack"
                ),
            }
        p = self.ws.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        sym = self._sem.find_symbol(p, symbol)
        if sym is None:
            raise ValueError(f"symbol {symbol!r} not found in {p.name}")
        text = p.read_bytes()[sym.start_byte:sym.end_byte].decode("utf-8", "replace")
        return {
            "path": str(p.relative_to(self.ws.target_dir)),
            "symbol": sym.qualified_name,
            "kind": sym.kind,
            "lines": [sym.start_line, sym.end_line],
            "source": text,
        }

    # ── observability ────────────────────────────────────────────────────

    def repo_map(
        self, path: str = ".", max_files: int = 80, max_symbols_per_file: int = 12,
    ) -> dict[str, Any]:
        """Compact outline of the project's key symbols (tree-sitter)."""
        if not Semantic.available():
            return {
                "error": "tree-sitter not available — install tree-sitter-language-pack",
            }
        root = self.ws.resolve(path)
        return self._sem.repo_map(
            root,
            max_files=max_files,
            max_symbols_per_file=max_symbols_per_file,
        )

    # ── secrets scan ─────────────────────────────────────────────────────

    def scan_secrets(self, path: str | None = None) -> dict[str, Any]:
        """Scan a file (or every text file under root) for likely credentials.

        Detects AWS keys, GitHub PATs, OpenAI / Anthropic API keys, JWTs,
        private-key blocks, etc.  Use before ``git_commit`` to avoid leaking
        credentials.
        """
        if path:
            p = self.ws.resolve(path)
            if not p.is_file():
                raise FileNotFoundError(str(p))
            return {
                "path": str(p.relative_to(self.ws.target_dir)),
                "hits": [
                    {"kind": h.kind, "line": h.line, "snippet": h.snippet}
                    for h in scan_secrets(p.read_text(encoding="utf-8", errors="replace"))
                ],
            }
        # Whole-repo scan: skip the same noise dirs grep does.
        all_hits: list[dict] = []
        for dirpath, dirnames, filenames in os.walk(self.ws.target_dir):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    if fp.stat().st_size > _MAX_FILE_SIZE:
                        continue
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError):
                    continue
                hits = scan_secrets(text)
                if hits:
                    rel = fp.relative_to(self.ws.target_dir)
                    for h in hits:
                        all_hits.append({
                            "path": str(rel),
                            "kind": h.kind,
                            "line": h.line,
                            "snippet": h.snippet,
                        })
                if len(all_hits) >= 50:
                    return {"hits": all_hits, "truncated": True}
        return {"hits": all_hits, "truncated": False}

    # ── embedding-based semantic search ─────────────────────────────────

    def semantic_search(
        self, query: str, k: int = 8, path_glob: str | None = None,
    ) -> dict[str, Any]:
        """Find code chunks *semantically* similar to *query*.

        Unlike ``grep`` (which needs literal matches), this surfaces code
        whose **meaning** matches the query — e.g. asking for "retry logic"
        finds an exponential-backoff loop even if the word "retry" doesn't
        appear.

        Auto-builds the index on first use.  Subsequent calls reuse it and
        only re-embed files whose ``(mtime, size)`` changed.
        """
        if not self._embed.available():
            return {
                "error": (
                    "embeddings not configured; set OPENAI_API_KEY (or "
                    "ADA_EMBED_API_KEY), or ADA_EMBED_FAKE=1 for offline tests"
                ),
            }
        # Lazy bootstrap: if there are no chunks yet, build the index.
        stats = self._embed.stats()
        if stats["chunks"] == 0:
            self._embed.index()
        hits = self._embed.search(query, k=k, path_glob=path_glob)
        return {"query": query, "k": k, "hits": hits}

    def reindex_embeddings(
        self, paths: list[str] | None = None, force: bool = False,
    ) -> dict[str, Any]:
        """Refresh the embedding index for *paths* (or whole repo).

        Call after large refactors or whenever ``semantic_search`` returns
        stale results.  ``force=True`` re-embeds even unchanged files.
        """
        if not self._embed.available():
            return {"error": "embeddings not configured"}
        return self._embed.index(paths=paths, force=force)

    def embed_stats(self) -> dict[str, Any]:
        """Report the embedding index state (chunk/file counts, model)."""
        return {
            **self._embed.stats(),
            "available": self._embed.available(),
        }

    # ── MCP (Model Context Protocol) servers ───────────────────────────

    def mcp_status(self) -> dict[str, Any]:
        """List configured MCP servers and the tools each one exposes."""
        return {
            "servers": [
                {
                    "name": name,
                    "tool_count": len(srv.tools),
                    "tools": [t.get("name") for t in srv.tools],
                }
                for name, srv in self._mcp.servers.items()
            ],
            "errors": list(self._mcp.errors),
            "ada_tool_names": sorted(self._mcp.tool_map.keys()),
        }

    # ── failure-locator + batch edit ───────────────────────────────────

    def locate_failures(self, traceback_text: str, max_frames: int = 20) -> dict[str, Any]:
        """Parse a pytest/unittest traceback into actionable frames.

        Returns each in-workspace frame as ``{file, line, symbol}`` plus a
        small source preview (±5 lines) so the model can fix the bug
        without an extra ``read_file`` round trip.
        """
        frames = in_workspace(parse_traceback(traceback_text), self.ws.target_dir)
        out: list[dict[str, Any]] = []
        for f in frames[:max_frames]:
            preview = ""
            try:
                p = self.ws.target_dir / f.file
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                lo = max(1, f.line - 5)
                hi = min(len(lines), f.line + 5)
                preview = "\n".join(
                    f"{i:>5}: {lines[i - 1]}" for i in range(lo, hi + 1)
                )
            except OSError:
                preview = "(source not readable)"
            out.append({**f.as_dict(), "preview": preview})
        return {"frames": out, "count": len(out)}

    def batch_edit(self, edits: list[dict[str, Any]], allow_secrets: bool = False) -> dict[str, Any]:
        """Apply many ``edit_file`` operations in one call.

        Each entry: ``{"path": str, "old": str, "new": str}``.  Edits run
        sequentially and the call is **transactional**: on any failure all
        previously-applied edits are rolled back to their original content.
        """
        backups: list[tuple[Path, str]] = []
        applied: list[dict[str, Any]] = []
        try:
            for i, e in enumerate(edits):
                path = e.get("path", "")
                old = e.get("old", "")
                new = e.get("new", "")
                p = self.ws.resolve(path)
                if not p.is_file():
                    raise FileNotFoundError(f"edit #{i}: {p}")
                backups.append((p, p.read_text(encoding="utf-8")))
                guarded = self._guard_write(new, allow_secrets)
                if guarded is not None:
                    raise ValueError(f"edit #{i}: {guarded.get('error')}")
                text = backups[-1][1]
                count = text.count(old)
                if count == 0:
                    raise ValueError(f"edit #{i}: old string not found in {path}")
                if count > 1:
                    raise ValueError(
                        f"edit #{i}: old string ambiguous ({count}x) in {path}"
                    )
                p.write_text(text.replace(old, new, 1), encoding="utf-8")
                applied.append({"path": path, "replaced": 1})
            return {"applied": applied, "count": len(applied)}
        except Exception as exc:
            # Rollback every backup we captured.
            for p, original in backups:
                try:
                    p.write_text(original, encoding="utf-8")
                except OSError:
                    pass
            return {"error": f"{type(exc).__name__}: {exc}", "rolled_back": len(backups)}

    # ── checkpoints (filesystem snapshot/rollback) ─────────────────────

    def create_checkpoint(self, label: str = "", paths: list[str] | None = None) -> dict[str, Any]:
        """Snapshot text files for later rollback.

        ``paths`` lets you scope the snapshot; omit to capture the whole
        workspace (still skipping ``.git``, ``.ada``, ``.venv``, etc).
        """
        cp = self._checkpoints.create(paths=paths, label=label)
        return {"id": cp.id, "files": len(cp.files)}

    def restore_checkpoint(self, id: str) -> dict[str, Any]:
        """Roll every captured file in checkpoint *id* back to its snapshot."""
        return self._checkpoints.restore(id)

    def list_checkpoints(self) -> dict[str, Any]:
        """Return all saved checkpoints, newest first."""
        return {"checkpoints": self._checkpoints.list()}

    # ── diff preview (read-only) ───────────────────────────────────────

    def preview_diff(self, ref: str = "HEAD", paths: list[str] | None = None) -> dict[str, Any]:
        """Show working-tree diff vs *ref* without committing or staging.

        Useful before ``git_commit`` to audit exactly what's about to land.
        Falls back to a "no git" notice when the workspace isn't a repo.
        """
        if not self.git.is_repo():
            return {"error": "not a git repository"}
        return {"ref": ref, "diff": self.git.diff(paths=paths, staged=False)}

    # ── self-introspection ─────────────────────────────────────────────

    def ada_config(self) -> dict[str, Any]:
        """Report Ada's current configuration: tool count, env caps, paths.

        Lets the agent introspect what's enabled (audit, secret guard,
        budget caps, embedding/MCP availability) without grepping env
        manually.  Read-only.
        """
        return {
            "target_dir": str(self.ws.target_dir),
            "tool_count": len(TOOL_SCHEMAS),
            "mcp_tool_count": len(self._mcp.tool_map),
            "embed_available": self._embed.available(),
            "env": {
                "ADA_AUDIT": os.getenv("ADA_AUDIT", "1"),
                "ADA_GUARD_SECRETS": os.getenv("ADA_GUARD_SECRETS", "1"),
                "ADA_MAX_COST": os.getenv("ADA_MAX_COST"),
                "ADA_MAX_STEPS_HARD": os.getenv("ADA_MAX_STEPS_HARD"),
                "ADA_MAX_TOKENS": os.getenv("ADA_MAX_TOKENS"),
                "ADA_MAX_SECONDS": os.getenv("ADA_MAX_SECONDS"),
                "ADA_PLAN_MODE": os.getenv("ADA_PLAN_MODE"),
            },
            "read_cache_size": len(self._read_cache),
            "read_cache_hits": self._read_cache_hits,
        }

    def make_plan(self, goal: str, context: str = "") -> dict[str, Any]:
        """Ask the planner LLM for a structured execution plan.

        Result is also persisted to ``.ada/plan.md`` so subsequent steps
        (and human reviewers) can see what Ada committed to.  This is a
        thin wrapper over ``consult_planner`` with a planning-specific
        rubric and an artifact write — separated out to make it easy for
        the agent to start a complex task with explicit planning.
        """
        if self._planner is None:
            return {"error": "no planner model configured"}
        prompt = (
            "Produce a concrete execution plan as Markdown.  Sections:\n"
            "  ## Goal (one sentence)\n"
            "  ## Assumptions (numbered, falsifiable)\n"
            "  ## Steps (numbered, each ≤1 line, mention the tool to use)\n"
            "  ## Risks & rollback strategy\n"
            "  ## Definition of done (verifiable)\n\n"
            f"GOAL: {goal}\n\n"
            f"CONTEXT:\n{context}\n"
        )
        try:
            plan = self._planner(prompt)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        # Persist to artifact (best-effort; never fail the tool because of FS).
        try:
            (self.ws.target_dir / ".ada").mkdir(exist_ok=True)
            (self.ws.target_dir / ".ada" / "plan.md").write_text(plan, encoding="utf-8")
        except OSError:
            pass
        return {"plan": plan, "saved_to": ".ada/plan.md"}

    def safe_run(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a mutating tool inside an auto-checkpoint.

        Snapshots the workspace, dispatches the inner tool, and rolls back
        if the tool returns ``{"error": ...}`` or raises.  Useful for
        speculative refactors that might break things.

        Returns ``{"ok": bool, "result": ..., "checkpoint": <id>, "rolled_back": bool}``.
        """
        args = args or {}
        fn = getattr(self, tool, None)
        if fn is None or tool in {"safe_run", "ada_config"}:
            return {"error": f"unknown or non-runnable tool {tool!r}"}
        cp = self._checkpoints.create(label=f"safe_{tool}")
        try:
            result = fn(**args)
        except Exception as exc:
            self._checkpoints.restore(cp.id)
            return {
                "ok": False,
                "checkpoint": cp.id,
                "rolled_back": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        # Soft rollback when the inner tool reports a structured error.
        is_err = isinstance(result, dict) and "error" in result
        if is_err:
            self._checkpoints.restore(cp.id)
        return {
            "ok": not is_err,
            "checkpoint": cp.id,
            "rolled_back": is_err,
            "result": result,
        }

    # ── refactor / focused tests / web ─────────────────────────────────

    def rename_symbol(
        self,
        old: str,
        new: str,
        dry_run: bool = False,
        extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Whole-word identifier rename across the workspace.

        Use ``dry_run=True`` first to inspect the hit list before
        committing to the change.  Refuses non-identifier inputs.
        """
        return _rename_symbol(self.ws.target_dir, old, new, dry_run=dry_run, extensions=extensions)

    def run_focused_tests(self, paths: list[str] | None = None, extra: str = "") -> dict[str, Any]:
        """Run pytest on a focused subset (changed files or explicit paths).

        When ``paths`` is omitted Ada queries ``git diff --name-only HEAD``
        to find modified ``test_*.py`` / ``*_test.py`` files.  Falls back
        to a no-op summary if nothing matches.
        """
        if not paths:
            try:
                changed = self.git.changed_files()
            except (GitError, AttributeError):
                changed = []
            paths = [
                p for p in changed
                if p.endswith(".py") and (
                    Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py")
                )
            ]
        if not paths:
            return {"skipped": True, "reason": "no changed test files detected"}
        cmd = "pytest -x -q " + " ".join(paths) + (f" {extra}" if extra else "")
        return self.run_command(cmd)

    def web_fetch(self, url: str, max_bytes: int = 200000) -> dict[str, Any]:
        """Fetch a documentation URL (HTTPS by default, allowlist via env).

        HTML is reduced to plain text; binary/long bodies are truncated.
        See ``ada.web`` for env knobs (``ADA_FETCH_ALLOWLIST``,
        ``ADA_FETCH_ALLOW_HTTP``).
        """
        return _fetch_url(url, max_bytes=max_bytes)

    # ── impact analysis & diff stats ───────────────────────────────────

    def impact_analysis(self, files: list[str] | None = None) -> dict[str, Any]:
        """Find tests that import any of *files* (or git-changed files).

        Use to drive a tighter test loop: only run tests that could
        plausibly be affected by the current diff.
        """
        if files is None:
            try:
                files = self.git.changed_files()
            except (GitError, AttributeError):
                files = []
        return changed_test_targets(self.ws.target_dir, files)

    def diff_stats(self, ref: str = "HEAD") -> dict[str, Any]:
        """Summarise pending diff vs *ref* with a risk verdict.

        Returns per-file added/removed counts and a coarse rating
        ('safe' / 'review' / 'risky') the agent can use to decide how
        much testing/review the change deserves.
        """
        if not self.git.is_repo():
            return {"error": "not a git repository"}
        text = self.git.diff(staged=False) + "\n" + self.git.diff(staged=True)
        return diff_stats(text)

    # ── persistent task queue ──────────────────────────────────────────

    def task_list(self) -> dict[str, Any]:
        """Return the persisted task queue (oldest first)."""
        return {"tasks": self._tasks.list()}

    def task_add(self, title: str) -> dict[str, Any]:
        """Append a new pending task to the queue."""
        return self._tasks.add(title)

    def task_update(self, id: int, status: str) -> dict[str, Any]:
        """Set a task's status (pending|in_progress|done|blocked)."""
        return self._tasks.update(id, status)

    def task_remove(self, id: int) -> dict[str, Any]:
        """Delete a task from the queue."""
        return self._tasks.remove(id)

    # ── notebook (.ipynb) read & cell edit ─────────────────────────────

    def read_notebook(self, path: str) -> dict[str, Any]:
        """Return a notebook's cells as a flat list (source + brief outputs)."""
        return _nb_read(self.ws.resolve(path))

    def edit_notebook_cell(self, path: str, index: int, new_source: str) -> dict[str, Any]:
        """Replace a single cell's source; preserves type/outputs/metadata."""
        return _nb_edit(self.ws.resolve(path), index, new_source)

    # ── persistent K/V memory (SQLite, cross-run) ──────────────────────

    def memory_set(self, ns: str, key: str, value: str) -> dict[str, Any]:
        """Upsert ``(ns, key) -> value`` in the persistent store."""
        return self._memory.set(ns, key, value)

    def memory_get(self, ns: str, key: str) -> dict[str, Any]:
        """Read a value back; returns ``{"missing": True}`` if absent."""
        return self._memory.get(ns, key)

    def memory_list(self, ns: str | None = None) -> dict[str, Any]:
        """Enumerate stored entries (optionally filtered by namespace)."""
        return self._memory.list(ns)

    def memory_delete(self, ns: str, key: str) -> dict[str, Any]:
        """Remove one entry; idempotent."""
        return self._memory.delete(ns, key)

    # ── profiling, PR drafting, flake hunting ──────────────────────────

    def profile_run(self, code: str, top: int = 15, sort: str = "cumulative") -> dict[str, Any]:
        """Run a Python snippet under cProfile; return the hottest funcs."""
        return _profile_python(code, top=top, sort=sort)

    def generate_pr_description(self, ref: str = "origin/main", max_diff_chars: int = 6000) -> dict[str, Any]:
        """Draft a PR description from ``git log`` + ``git diff`` vs *ref*.

        Requires a planner callable to be wired in (set on this Tools
        instance by the agent).  The result is also written to
        ``.ada/pr_description.md`` for inspection.
        """
        if not self.git.is_repo():
            return {"error": "not a git repository"}
        planner = getattr(self, "_planner", None)
        if planner is None:
            return {"error": "planner not configured"}
        try:
            log = self.git._git("log", f"{ref}..HEAD", "--pretty=format:%h %s")
        except GitError as exc:
            return {"error": f"git log failed: {exc}"}
        try:
            diff = self.git._git("diff", f"{ref}...HEAD")
        except GitError as exc:
            return {"error": f"git diff failed: {exc}"}
        if len(diff) > max_diff_chars:
            diff = diff[:max_diff_chars] + f"\n... <truncated {len(diff) - max_diff_chars} chars>\n"

        prompt = (
            "Write a concise pull-request description in markdown. Use these "
            "sections: ## Summary, ## Changes, ## Test plan, ## Risk. "
            "Be terse and specific.\n\n"
            f"### git log\n{log or '(no commits ahead of ref)'}\n\n"
            f"### git diff (truncated)\n```diff\n{diff}\n```\n"
        )
        try:
            text = planner(prompt) or ""
        except Exception as exc:  # noqa: BLE001
            return {"error": f"planner failed: {type(exc).__name__}: {exc}"}
        out_path = self.ws.target_dir / ".ada" / "pr_description.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return {"description": text, "path": str(out_path), "ref": ref}

    def flake_check(self, paths: list[str] | None = None, runs: int = 3, extra: str = "") -> dict[str, Any]:
        """Run pytest *runs* times on *paths*; report per-run pass/fail.

        A test that flips outcome between runs is flagged 'flaky'.
        """
        if runs < 2:
            return {"error": "runs must be >= 2 to detect flakiness"}
        scope = " ".join(paths) if paths else ""
        cmd_extra = (" " + extra.strip()) if extra.strip() else ""
        results: list[dict] = []
        for i in range(runs):
            r = self.run_command(f"pytest -q {scope}{cmd_extra}".strip())
            results.append({
                "run": i + 1,
                "exit_code": r.get("exit_code"),
                "ok": r.get("exit_code") == 0,
            })
        oks = {r["ok"] for r in results}
        verdict = "stable" if len(oks) == 1 else "flaky"
        return {
            "runs": results,
            "verdict": verdict,
            "all_passed": all(r["ok"] for r in results),
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
            "description": "Create or overwrite a file with new content. Use sparingly; prefer edit_file for surgical changes. Refuses to write content matching credential patterns unless allow_secrets=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "allow_secrets": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one occurrence of `old` with `new` in a file. Fails if `old` is missing or ambiguous. Include enough context in `old` to be unique. Refuses to write `new` content matching credential patterns unless allow_secrets=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "allow_secrets": {"type": "boolean", "default": False},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ast_edit",
            "description": (
                "Structural Python edit by symbol name (zero-dep, stdlib `ast`). "
                "Prefer this over `edit_file` for whole-function/class rewrites — it "
                "is robust to formatting drift and validates that the result still parses. "
                "Operations: 'show' | 'replace' | 'delete' | 'insert_before' | 'insert_after' | 'append'. "
                "Target syntax: 'foo' for top-level def/class, 'MyClass.bar' for methods (one level deep). "
                "`code` is required for replace/insert_*/append and must be valid Python; it is auto-reindented "
                "to match the target's column. 'append' ignores `target` and adds `code` at end of file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "show",
                            "replace",
                            "delete",
                            "insert_before",
                            "insert_after",
                            "append",
                        ],
                    },
                    "target": {"type": "string", "default": ""},
                    "code": {"type": "string", "default": ""},
                },
                "required": ["path", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_apply",
            "description": (
                "Apply a unified-diff patch (`diff -u` / `git diff` format) across "
                "one or more files atomically. Each file section starts with `--- <old>` "
                "and `+++ <new>` (any `a/`/`b/` prefixes stripped); use `/dev/null` to "
                "create or delete a file. Hunks are matched by content (the @@ line "
                "numbers are only a hint; the `-`/context block must occur in the file, "
                "with ±200 lines of drift tolerated). If ANY hunk fails, NO files are "
                "written. Set `check_only` for a dry run. Prefer this over many "
                "`edit_file`/`ast_edit` calls when changing several places at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_symbols",
            "description": (
                "List every named definition (function/class/method/type/etc.) in a "
                "source file via tree-sitter. Supports py, js, ts, tsx, go, rs, java, "
                "c, cpp. Use this instead of read_file when you only need a structural "
                "overview of what's in a file. Each symbol carries a `qualified_name` "
                "(`Class.method` for nested defs) usable directly with `ts_edit`."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": (
                "Project-wide: locate every definition whose leaf name equals `name` "
                "(tree-sitter, multi-language). Returns file/line/qualified_name for "
                "each hit. Far more precise than grep — only real definitions, not "
                "occurrences inside strings/comments/calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": (
                "Project-wide: identifier-position occurrences of `name` "
                "(tree-sitter). Excludes string literals and comments automatically. "
                "NOT scope-aware: a local variable shadowing the name is still "
                "reported. For navigating a codebase or planning a rename, follow up "
                "by reading the hit context with `read_file`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 500},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ts_edit",
            "description": (
                "Multi-language structural edit by symbol name (tree-sitter). "
                "Same operations as ast_edit — show / replace / delete / insert_before / "
                "insert_after — but works on JS/TS/Go/Rust/Java/C/C++ in addition to "
                "Python. The new code is auto-reindented to the symbol's column and the "
                "result is re-parsed: if it introduces NEW parse errors the edit is "
                "rolled back. Use `ast_edit` for Python (faster, supports append); use "
                "`ts_edit` for every other language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "show", "replace", "delete",
                            "insert_before", "insert_after",
                        ],
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Dotted qualified name, e.g. 'Foo' or 'Foo.bar'.",
                    },
                    "code": {"type": "string", "default": ""},
                },
                "required": ["path", "operation", "symbol"],
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
            "description": "Run a shell command inside the target project. Use for: install deps, build, run tests, lint, benchmarks. stdout/stderr are tail-truncated. Dangerous patterns (rm -rf, git push --force, curl|sh, ...) are blocked unless `force=True`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Bypass the safety guard for this call only. Get human OK first.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Auto-detect and run the project's test suite(s) — pytest, unittest, "
                "go test, cargo test, npm test. Returns per-tool exit codes and "
                "tail-truncated stdout/stderr so you can react to failures. Pass "
                "`only=['pytest']` to limit. Always run after non-trivial edits before commit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "only": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of detected test tools to run; default: all detected.",
                    },
                    "timeout": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_lint",
            "description": (
                "Auto-detect and run lint / typecheck tools — ruff, mypy, eslint, tsc, "
                "go vet, clippy. Same shape as run_tests. Use after edits to catch "
                "regressions the parser-level ts_edit/ast_edit checks can't see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "only": {"type": "array", "items": {"type": "string"}},
                    "timeout": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_toolchain",
            "description": "Quick: list which test/lint/format tools were detected in this repo.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_symbol",
            "description": (
                "Return the source of a single function/class/method by qualified "
                "name. ~10× cheaper than read_file when you only need one symbol — "
                "ideal after list_symbols / find_definition has located the target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbol": {"type": "string", "description": "Dotted name, e.g. 'Foo.bar'."},
                },
                "required": ["path", "symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": (
                "Compact outline of the project's key symbols (tree-sitter). Use it "
                "before deep work to orient yourself instead of multiple list_dir + "
                "read_file calls. Files are ranked: shallower paths and richer "
                "files come first; tests are de-prioritised."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_files": {"type": "integer", "default": 80},
                    "max_symbols_per_file": {"type": "integer", "default": 12},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_secrets",
            "description": (
                "Scan a file (or the whole repo) for likely credential leaks — AWS "
                "keys, GitHub PATs, OpenAI / Anthropic API keys, JWTs, private-key "
                "blocks. ALWAYS run before git_commit on changed files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Single file to scan; omit to scan the whole repo.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Embedding-based code search. Finds chunks whose MEANING matches "
                "the query, not just literal keywords. Use this BEFORE grep when "
                "exploring an unfamiliar codebase or when you don't know the "
                "exact identifier names. Auto-builds the index on first call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description, e.g. 'where is the retry/backoff logic'.",
                    },
                    "k": {"type": "integer", "default": 8},
                    "path_glob": {
                        "type": "string",
                        "description": "Optional fnmatch pattern to restrict scope, e.g. 'src/**/*.py'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reindex_embeddings",
            "description": (
                "Refresh the embedding index. Call after large refactors. "
                "Pass paths=null to reindex the whole repo; force=true re-embeds "
                "even unchanged files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Repo-relative file paths to refresh; omit for full reindex.",
                    },
                    "force": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "embed_stats",
            "description": "Report embedding-index status (chunks, files, model, availability).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_status",
            "description": (
                "List configured MCP (Model Context Protocol) servers and the "
                "tools each one exposes. Use to discover which mcp_<server>_<tool> "
                "calls are available."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_failures",
            "description": (
                "Parse a pytest/unittest traceback and return the in-workspace "
                "frames (file, line, symbol) plus a ±5-line source preview, so "
                "you can fix bugs without extra read_file calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "traceback_text": {"type": "string"},
                    "max_frames": {"type": "integer", "default": 20},
                },
                "required": ["traceback_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_edit",
            "description": (
                "Apply many edit_file operations transactionally. On any "
                "failure all previously-applied edits in the batch are rolled "
                "back. Use this for cross-file renames or coordinated changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                            },
                            "required": ["path", "old", "new"],
                        },
                    },
                    "allow_secrets": {"type": "boolean", "default": False},
                },
                "required": ["edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_checkpoint",
            "description": (
                "Snapshot text files for later rollback. Returns a checkpoint id. "
                "Use before risky multi-file rewrites; pair with restore_checkpoint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of paths to snapshot; omit for whole workspace.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restore_checkpoint",
            "description": "Roll every captured file in checkpoint `id` back to its snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_checkpoints",
            "description": "Return all saved checkpoints, newest first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_diff",
            "description": (
                "Show working-tree diff vs `ref` (default HEAD) without staging or "
                "committing. Use before git_commit to audit pending changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "default": "HEAD"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ada_config",
            "description": (
                "Report Ada's current configuration: tool count, env caps "
                "(audit, secret guard, budget), embedding/MCP availability, "
                "read-cache stats. Read-only; useful for self-introspection."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_plan",
            "description": (
                "Ask the planner LLM for a structured execution plan "
                "(Goal/Assumptions/Steps/Risks/DoD) and persist it to "
                ".ada/plan.md. Use before tackling complex multi-step tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "safe_run",
            "description": (
                "Run a mutating tool inside an auto-checkpoint. Snapshots the "
                "workspace, dispatches the inner tool, and rolls back on error "
                "(exception or {error: ...} return). Use for speculative changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["tool"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_symbol",
            "description": (
                "Whole-word identifier rename across the workspace (textual, "
                "import-aware via word boundaries). Use dry_run=true first to "
                "audit hits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file extensions (e.g. ['.py']).",
                    },
                },
                "required": ["old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_focused_tests",
            "description": (
                "Run pytest only on changed test files (via git diff) or an "
                "explicit list. Faster feedback loop than full run_tests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "extra": {"type": "string", "description": "extra pytest args"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "GET a documentation URL (HTTPS by default). HTML is reduced "
                "to plain text. Honours ADA_FETCH_ALLOWLIST and "
                "ADA_FETCH_ALLOW_HTTP env vars. Use for API/docs lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 200000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impact_analysis",
            "description": (
                "Find tests that import the given changed files (or, when "
                "omitted, the current git diff). Use to scope the next "
                "run_tests/run_focused_tests call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_stats",
            "description": (
                "Summarise pending diff vs ref with per-file added/removed "
                "counts and a 'safe' / 'review' / 'risky' verdict."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "string", "default": "HEAD"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "Return the persisted multi-goal task queue (oldest first).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_add",
            "description": "Append a new pending task to the queue.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "Set a task's status: pending | in_progress | done | blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "status": {"type": "string"},
                },
                "required": ["id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_remove",
            "description": "Delete a task from the queue by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": "Return a Jupyter notebook's cells (source + brief outputs).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_notebook_cell",
            "description": "Replace a single notebook cell's source by index. Preserves outputs and metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "index": {"type": "integer"},
                    "new_source": {"type": "string"},
                },
                "required": ["path", "index", "new_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_set",
            "description": "Upsert (ns, key) → value in the persistent SQLite memory store. Survives across runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ns": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["ns", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get",
            "description": "Read one value from persistent memory. Returns {missing: true} if absent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ns": {"type": "string"},
                    "key": {"type": "string"},
                },
                "required": ["ns", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "List entries in persistent memory (optionally filtered by namespace).",
            "parameters": {
                "type": "object",
                "properties": {"ns": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Remove one (ns, key) entry from persistent memory. Idempotent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ns": {"type": "string"},
                    "key": {"type": "string"},
                },
                "required": ["ns", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_run",
            "description": "Run a Python snippet under cProfile; return the hottest functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "top": {"type": "integer", "default": 15},
                    "sort": {"type": "string", "default": "cumulative"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pr_description",
            "description": "Draft a PR description (markdown) from git log + diff vs ref. Requires planner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "default": "origin/main"},
                    "max_diff_chars": {"type": "integer", "default": 6000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flake_check",
            "description": "Run pytest N times on the given paths to detect flaky tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "runs": {"type": "integer", "default": 3},
                    "extra": {"type": "string", "default": ""},
                },
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
