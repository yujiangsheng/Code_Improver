"""Multi-language semantic helpers backed by tree-sitter.

Why tree-sitter (not LSP)?
--------------------------
LSP would give us scope-perfect references and rename, but it requires a
running language server per language, JSON-RPC plumbing, and project-aware
configuration (compile_commands.json, tsconfig.json, ...).  tree-sitter is a
single ``pip install`` with prebuilt wheels for ~165 languages and gives us
syntactic-level navigation that already beats grep by a large margin:

    list_symbols      enumerate top-level + nested defs in a file
    find_symbol       resolve a dotted ``A.B.C`` qualified name → AST range
    find_definition   list every definition of a name across the project
    find_references   list every identifier-position occurrence of a name
                      (scope-unaware — see caveats below)
    edit_symbol       structural replace/delete/insert by symbol name,
                      multi-language, syntax-validated post-write

Caveats
-------
* ``find_references`` is **textual at identifier nodes**: it correctly skips
  occurrences inside strings, comments, attribute names of *other* objects,
  etc., but it cannot tell ``foo`` (the variable) from ``foo`` (an unrelated
  symbol shadowed in a nested scope).  For project-wide rename safety you
  still want LSP — this implementation is meant for guided exploration, not
  blind refactoring.
* ``edit_symbol`` validates the new file by re-parsing and counting
  ``ERROR`` / ``MISSING`` nodes; if the count goes up, the edit is rejected
  (the original file is left untouched).
* Languages without a registered parser fall back to "unknown" and the tools
  return a structured error so callers can fall back to ``grep``/``ast_edit``.

Optional dependency
-------------------
``tree_sitter_language_pack`` ships prebuilt parsers.  When the package is not
installed, ``Semantic.available()`` returns False and every operation returns
``{"error": "tree-sitter not available", ...}``; the rest of Ada keeps working.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:                                                   # pragma: no cover - optional
    from tree_sitter_language_pack import get_parser  # type: ignore
    _HAS_TS = True
except Exception:                                      # pragma: no cover
    get_parser = None                                  # type: ignore
    _HAS_TS = False


# ── language registry ────────────────────────────────────────────────────────
# Map file extension → (tree-sitter language name, definition node types).
# The "name field" used to read a node's identifier defaults to "name" when
# tree-sitter exposes it; for a few node types we read the first identifier
# child instead (declared per-entry).
@dataclass(frozen=True)
class _LangSpec:
    """Per-language definition-node descriptor.

    ``def_types``
        Set of tree-sitter node ``type`` strings that count as a "definition"
        (functions, classes, methods, types, etc.).  Anything outside this
        set is ignored when listing or resolving symbols.
    ``name_fields``
        Ordered list of strategies used to extract the identifier text from a
        definition node:
        * ``"field:NAME"`` — call ``child_by_field_name(NAME)``
        * ``"first_identifier"`` — first descendant of type ``identifier``
                                    (or ``type_identifier`` for Rust types)
    ``container_types``
        Node types whose body should be descended into when building a
        qualified name (e.g. classes contain methods).  Unset means
        ``def_types`` doubles as containers.
    """
    name: str
    def_types: tuple[str, ...]
    name_fields: tuple[str, ...] = ("field:name", "first_identifier")
    container_types: tuple[str, ...] = ()


_LANGS: dict[str, _LangSpec] = {
    ".py": _LangSpec(
        "python",
        def_types=(
            "function_definition", "class_definition",
            "decorated_definition",
        ),
        # decorated_definition is NOT a container: its only child is the
        # wrapped function/class, which we already named via _name_of().
        # Recursing into it would re-emit the inner def as a duplicate.
        container_types=("class_definition",),
    ),
    ".js": _LangSpec(
        "javascript",
        def_types=(
            "function_declaration", "class_declaration",
            "method_definition", "generator_function_declaration",
        ),
        container_types=("class_declaration",),
    ),
    ".mjs": _LangSpec("javascript", def_types=(
        "function_declaration", "class_declaration", "method_definition")),
    ".cjs": _LangSpec("javascript", def_types=(
        "function_declaration", "class_declaration", "method_definition")),
    ".jsx": _LangSpec("javascript", def_types=(
        "function_declaration", "class_declaration", "method_definition")),
    ".ts": _LangSpec(
        "typescript",
        def_types=(
            "function_declaration", "class_declaration",
            "method_definition", "interface_declaration",
            "type_alias_declaration", "enum_declaration",
        ),
        container_types=("class_declaration", "interface_declaration"),
    ),
    ".tsx": _LangSpec(
        "tsx",
        def_types=(
            "function_declaration", "class_declaration",
            "method_definition", "interface_declaration",
            "type_alias_declaration",
        ),
        container_types=("class_declaration", "interface_declaration"),
    ),
    ".go": _LangSpec(
        "go",
        def_types=("function_declaration", "method_declaration", "type_declaration"),
    ),
    ".rs": _LangSpec(
        "rust",
        def_types=(
            "function_item", "struct_item", "enum_item",
            "trait_item", "impl_item", "mod_item",
        ),
        name_fields=("field:name", "first_identifier"),
        container_types=("impl_item", "trait_item", "mod_item"),
    ),
    ".java": _LangSpec(
        "java",
        def_types=(
            "method_declaration", "class_declaration",
            "interface_declaration", "constructor_declaration",
            "enum_declaration",
        ),
        container_types=("class_declaration", "interface_declaration", "enum_declaration"),
    ),
    ".c": _LangSpec("c", def_types=("function_definition",)),
    ".h": _LangSpec("c", def_types=("function_definition",)),
    ".cpp": _LangSpec(
        "cpp",
        def_types=("function_definition", "class_specifier", "struct_specifier"),
        container_types=("class_specifier", "struct_specifier"),
    ),
    ".cc": _LangSpec("cpp", def_types=("function_definition",)),
    ".cxx": _LangSpec("cpp", def_types=("function_definition",)),
    ".hpp": _LangSpec("cpp", def_types=("function_definition", "class_specifier")),
    ".hh":  _LangSpec("cpp", def_types=("function_definition", "class_specifier")),
}


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    """A located definition in source code.

    ``qualified_name`` uses ``"."`` as the separator for nested scopes
    (e.g. ``MyClass.inner.method`` for a method inside a nested class).
    Line numbers are 1-indexed and **inclusive** — ``[start_line, end_line]``
    denotes the same span as ``ast_edit``.
    """
    name: str
    qualified_name: str
    kind: str           # tree-sitter node type
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


# ── core engine ──────────────────────────────────────────────────────────────

class Semantic:
    """Project-aware tree-sitter operations.

    All paths passed in are project-relative; the caller (``Tools``) resolves
    them through the workspace sandbox before handing them over.
    """

    # Directories to skip when crawling for definitions/references.
    _SKIP = {
        ".git", ".ada", "node_modules", ".venv", "venv", "__pycache__",
        "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".tox",
    }
    _MAX_FILE_SIZE = 1_000_000  # 1 MB cap, same as grep

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = Path(target_dir)

    # ── availability ─────────────────────────────────────────────────────

    @staticmethod
    def available() -> bool:
        """True when ``tree_sitter_language_pack`` imported successfully."""
        return _HAS_TS

    @staticmethod
    def supported_extensions() -> list[str]:
        """List of file extensions we know how to parse."""
        return sorted(_LANGS.keys())

    # ── parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _spec_for(path: Path) -> _LangSpec | None:
        """Return the language spec for *path* by extension, or None."""
        return _LANGS.get(path.suffix.lower())

    @staticmethod
    def _parse(spec: _LangSpec, source: bytes):
        """Parse *source* with the parser for *spec*; return the root node."""
        parser = get_parser(spec.name)  # type: ignore[misc]
        return parser.parse(source).root_node

    @staticmethod
    def _node_text(node, source: bytes) -> str:
        """Return ``node`` text decoded as UTF-8 (replace on error)."""
        return source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    @staticmethod
    def _name_of(node, spec: _LangSpec, source: bytes) -> str | None:
        """Extract the identifier text from a definition *node* per *spec*.

        Tries each strategy in ``spec.name_fields`` in order; returns None if
        nothing yields a name (e.g. anonymous functions or impl blocks).
        For Python's ``decorated_definition`` we descend into the wrapped
        function/class because the name lives there, not on the decorator.
        """
        # Decorated Python definitions: descend to the inner function/class.
        if node.type == "decorated_definition":
            for ch in node.children:
                if ch.type in ("function_definition", "class_definition"):
                    return Semantic._name_of(ch, spec, source)
            return None

        for strat in spec.name_fields:
            if strat.startswith("field:"):
                field = strat.split(":", 1)[1]
                child = node.child_by_field_name(field)
                if child is not None:
                    return Semantic._node_text(child, source)
            elif strat == "first_identifier":
                for ch in node.children:
                    if ch.type in ("identifier", "type_identifier",
                                   "field_identifier", "property_identifier"):
                        return Semantic._node_text(ch, source)
        return None

    # ── symbol enumeration ──────────────────────────────────────────────

    def list_symbols(self, path: str | Path) -> list[Symbol]:
        """Return every named definition in *path*, depth-first.

        Children inside container types (classes, traits, modules) are
        emitted with a dotted qualified name; leaf functions get their bare
        name as ``qualified_name``.
        """
        p = self._abs(path)
        spec = self._spec_for(p)
        if spec is None:
            raise ValueError(f"unsupported language for {p.name}")
        source = p.read_bytes()
        root = self._parse(spec, source)
        out: list[Symbol] = []
        self._walk_defs(root, spec, source, prefix="", out=out)
        return out

    def _walk_defs(
        self, node, spec: _LangSpec, source: bytes,
        prefix: str, out: list[Symbol],
    ) -> None:
        """Depth-first scan for definition nodes; recurse into containers."""
        for child in node.children:
            if child.type in spec.def_types:
                name = self._name_of(child, spec, source)
                if name:
                    qname = f"{prefix}.{name}" if prefix else name
                    out.append(Symbol(
                        name=name,
                        qualified_name=qname,
                        kind=child.type,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                    ))
                    # Recurse into containers (classes, impl blocks, ...).
                    container = (
                        spec.container_types
                        or spec.def_types  # default: every def can contain
                    )
                    if child.type in container:
                        self._walk_defs(child, spec, source, qname, out)
                    continue
            # Not a definition itself — descend so nested defs are found.
            self._walk_defs(child, spec, source, prefix, out)

    # ── symbol resolution / project-wide search ─────────────────────────

    def find_symbol(self, path: str | Path, qualified_name: str) -> Symbol | None:
        """Look up *qualified_name* in *path*; return the matching ``Symbol``."""
        for sym in self.list_symbols(path):
            if sym.qualified_name == qualified_name:
                return sym
        return None

    def find_definition(self, name: str, root: str | Path = ".") -> list[dict]:
        """Project-wide: every definition whose **leaf name** equals *name*.

        Matches both bare ``foo`` and any ``Class.foo`` / ``mod::foo`` —
        i.e. any symbol whose last dotted segment is *name*.
        """
        results: list[dict] = []
        for f in self._iter_source_files(self._abs(root)):
            try:
                syms = self.list_symbols(f)
            except Exception:
                continue
            for s in syms:
                if s.name == name:
                    results.append({
                        "path": str(f.relative_to(self.target_dir)),
                        "qualified_name": s.qualified_name,
                        "kind": s.kind,
                        "line": s.start_line,
                    })
        return results

    def find_references(
        self, name: str, root: str | Path = ".", max_results: int = 500,
    ) -> list[dict]:
        """Project-wide: identifier-position occurrences of *name*.

        Walks the parse tree of every supported file and yields every
        ``identifier`` (or language-equivalent) node whose text equals *name*.
        This automatically excludes occurrences inside string literals and
        comments.  It does NOT do scope analysis: a local variable shadowing
        the name will still be reported.

        Returns at most *max_results* hits; the result dict ``truncated`` flag
        signals when the cap was reached.
        """
        hits: list[dict] = []
        for f in self._iter_source_files(self._abs(root)):
            spec = self._spec_for(f)
            if spec is None:
                continue
            try:
                source = f.read_bytes()
                tree_root = self._parse(spec, source)
            except Exception:
                continue
            for node in self._walk_identifiers(tree_root):
                txt = self._node_text(node, source)
                if txt == name:
                    line_no = node.start_point[0] + 1
                    line_text = source.split(b"\n")[node.start_point[0]]
                    hits.append({
                        "path": str(f.relative_to(self.target_dir)),
                        "line": line_no,
                        "col": node.start_point[1] + 1,
                        "text": line_text.decode("utf-8", "replace").strip(),
                    })
                    if len(hits) >= max_results:
                        return hits
        return hits

    @staticmethod
    def _walk_identifiers(node) -> Iterable:
        """Yield every identifier-like leaf node in the subtree."""
        if not node.children:
            if node.type in (
                "identifier", "type_identifier", "field_identifier",
                "property_identifier", "shorthand_property_identifier",
            ):
                yield node
            return
        for ch in node.children:
            yield from Semantic._walk_identifiers(ch)

    # ── repo map ────────────────────────────────────────────────────────

    def repo_map(
        self,
        root: str | Path = ".",
        max_files: int = 80,
        max_symbols_per_file: int = 12,
    ) -> dict:
        """Build a compact, prompt-friendly outline of the project.

        For each supported file under *root* we list its top-level symbols
        (functions / classes / types) together with their line ranges.  The
        result is intentionally small enough to inject into the system prompt
        on every request so the model has constant awareness of where things
        live without burning tokens on directory walks.

        Files / symbols are ranked: source files at shallower depth, with
        more symbols, win.  Tests (paths containing ``/test``) are
        de-prioritised.
        """
        root_abs = self._abs(root)
        ranked: list[tuple[float, Path, list[Symbol]]] = []
        for f in self._iter_source_files(root_abs):
            try:
                syms = self.list_symbols(f)
            except Exception:
                continue
            if not syms:
                continue
            try:
                rel = f.relative_to(self.target_dir)
            except ValueError:
                rel = f
            depth = len(rel.parts)
            test_penalty = 5 if any("test" in part.lower() for part in rel.parts) else 0
            # Higher score = more important; we sort descending later.
            score = float(len(syms)) - depth - test_penalty
            ranked.append((score, f, syms))

        ranked.sort(key=lambda t: t[0], reverse=True)
        ranked = ranked[:max_files]

        files_out: list[dict] = []
        for _, f, syms in ranked:
            try:
                rel = f.relative_to(self.target_dir)
            except ValueError:
                rel = f
            # Keep top-level symbols (no dot) plus a few notable nested ones.
            top = [s for s in syms if "." not in s.qualified_name]
            top.sort(key=lambda s: s.start_line)
            entries: list[dict] = []
            for s in top[:max_symbols_per_file]:
                entries.append({
                    "name": s.qualified_name,
                    "kind": s.kind,
                    "lines": [s.start_line, s.end_line],
                })
            files_out.append({
                "path": str(rel),
                "language": f.suffix.lstrip("."),
                "symbols": entries,
                "symbol_count": len(syms),
            })
        return {
            "root": str(root_abs.relative_to(self.target_dir))
                    if root_abs != self.target_dir else ".",
            "files": files_out,
            "total_files_scanned": len(ranked),
        }

    @staticmethod
    def render_repo_map(rmap: dict, max_chars: int = 6000) -> str:
        """Render :py:meth:`repo_map` as compact markdown for prompt injection."""
        lines: list[str] = []
        lines.append(f"# Repo map ({len(rmap['files'])} key files)")
        for entry in rmap["files"]:
            sym_count = entry["symbol_count"]
            shown = len(entry["symbols"])
            extra = (
                f"  ({sym_count - shown} more symbols)"
                if sym_count > shown else ""
            )
            lines.append(f"\n**{entry['path']}**{extra}")
            for s in entry["symbols"]:
                a, b = s["lines"]
                lines.append(f"- {s['name']}  *{s['kind']}*  L{a}-{b}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n…[truncated]"
        return out

    # ── structural editing ──────────────────────────────────────────────

    def edit_symbol(
        self,
        path: str | Path,
        qualified_name: str,
        operation: str,
        code: str = "",
    ) -> dict:
        """Replace / delete / insert / show a symbol's definition span.

        Operations mirror :py:meth:`Tools.ast_edit` but work for every
        registered language.  After writing, the file is re-parsed; if the
        edit increased the count of ``ERROR``/``MISSING`` parse nodes, the
        write is rolled back and ``ValueError`` is raised.

        Returns a dict describing the action taken.  ``operation="show"``
        also includes ``"source"``.
        """
        p = self._abs(path)
        spec = self._spec_for(p)
        if spec is None:
            raise ValueError(f"unsupported language for {p.name}")
        original = p.read_bytes()
        baseline_errors = self._count_errors(self._parse(spec, original))

        sym = self.find_symbol(p, qualified_name)
        if sym is None:
            # Help the model self-correct: surface the closest existing
            # symbols by leaf-name and qualified-name similarity.
            try:
                all_syms = self.list_symbols(p)
            except Exception:
                all_syms = []
            suggestions = self._suggest(qualified_name, all_syms, k=5)
            hint = (
                f"; did you mean: {', '.join(suggestions)}"
                if suggestions else ""
            )
            raise ValueError(
                f"symbol {qualified_name!r} not found in {p.name}{hint}"
            )

        start_b, end_b = sym.start_byte, sym.end_byte
        text = original.decode("utf-8")

        if operation == "show":
            return {
                "path": str(p.relative_to(self.target_dir)),
                "operation": operation,
                "qualified_name": qualified_name,
                "kind": sym.kind,
                "lines": [sym.start_line, sym.end_line],
                "source": original[start_b:end_b].decode("utf-8", "replace"),
            }

        # All remaining operations mutate the file.
        if operation == "delete":
            new_text = self._splice(text, start_b, end_b, "")
        elif operation in ("replace", "insert_before", "insert_after"):
            if not code.strip():
                raise ValueError(f"code is required for operation {operation!r}")
            # Determine the column of the first byte of the symbol on its
            # start line so the inserted block lines up with the original.
            line_start = original.rfind(b"\n", 0, start_b) + 1
            col = start_b - line_start
            indent = " " * col

            if operation == "replace":
                # The splice begins exactly at the symbol's column, so the
                # FIRST line must not be re-indented; only lines 2..N do.
                block = self._reindent_tail(code, indent)
                new_text = self._splice(text, start_b, end_b, block)
            elif operation == "insert_before":
                # Same first-line rule as replace; then a blank line + the
                # leading whitespace re-supplied so the original's first
                # line keeps its indent.
                block = self._reindent_tail(code, indent)
                new_text = self._splice(
                    text, start_b, start_b, block + "\n\n" + indent
                )
            else:  # insert_after
                # We splice at end_byte, which is typically right after the
                # last token of the symbol (column 0 follows the next \n).
                # Use a blank line + fully-indented block so the new
                # definition starts at the same column as the target.
                block = self._reindent(code, indent)
                new_text = self._splice(
                    text, end_b, end_b, "\n\n" + block
                )
        else:
            raise ValueError(f"unknown operation: {operation!r}")

        # Validate by re-parsing.
        new_root = self._parse(spec, new_text.encode("utf-8"))
        new_errors = self._count_errors(new_root)
        if new_errors > baseline_errors:
            raise ValueError(
                f"edit would introduce {new_errors - baseline_errors} new parse "
                f"error(s); aborted (file unchanged)"
            )

        p.write_text(new_text, encoding="utf-8")
        return {
            "path": str(p.relative_to(self.target_dir)),
            "operation": operation,
            "qualified_name": qualified_name,
            "kind": sym.kind,
            "lines": [sym.start_line, sym.end_line],
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _abs(self, path: str | Path) -> Path:
        """Resolve a workspace-relative path to an absolute Path."""
        p = Path(path)
        if not p.is_absolute():
            p = self.target_dir / p
        return p

    def _iter_source_files(self, root: Path) -> Iterable[Path]:
        """Yield every supported source file under *root*, skipping noise dirs."""
        if root.is_file():
            if root.suffix.lower() in _LANGS:
                yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self._SKIP]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in _LANGS:
                    continue
                try:
                    if p.stat().st_size <= self._MAX_FILE_SIZE:
                        yield p
                except OSError:
                    continue

    @staticmethod
    def _splice(text: str, start_b: int, end_b: int, repl: str) -> str:
        """Byte-aware replacement: preserves multibyte characters around the cut."""
        b = text.encode("utf-8")
        return (b[:start_b] + repl.encode("utf-8") + b[end_b:]).decode("utf-8")

    @staticmethod
    def _suggest(query: str, syms: list, k: int = 5) -> list[str]:
        """Closest existing qualified names to *query* (difflib-based)."""
        import difflib
        if not syms:
            return []
        # Score against both leaf and qualified names; keep the higher of the two.
        names = []
        leaf = query.rsplit(".", 1)[-1]
        for s in syms:
            qname = s.qualified_name
            score = max(
                difflib.SequenceMatcher(None, leaf, s.name).ratio(),
                difflib.SequenceMatcher(None, query, qname).ratio(),
            )
            names.append((score, qname))
        names.sort(key=lambda t: t[0], reverse=True)
        return [n for score, n in names[:k] if score >= 0.4]

    @staticmethod
    def _count_errors(root) -> int:
        """Recursively count ``ERROR`` and ``MISSING`` nodes in a tree."""
        n = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_error or node.is_missing:
                n += 1
            stack.extend(node.children)
        return n

    @staticmethod
    def _reindent(code: str, indent: str) -> str:
        """Same dedent + re-indent pass used by ``Tools.ast_edit``."""
        import textwrap
        dedented = textwrap.dedent(code).rstrip("\n")
        if not indent:
            return dedented
        return "\n".join(
            (indent + ln) if ln.strip() else ln
            for ln in dedented.splitlines()
        )

    @staticmethod
    def _reindent_tail(code: str, indent: str) -> str:
        """Like :py:meth:`_reindent` but leaves the **first line** untouched.

        Used for ``replace`` and ``insert_before`` where the splice point is
        already at the desired column — adding indent to the first line
        would double up.
        """
        import textwrap
        dedented = textwrap.dedent(code).rstrip("\n")
        lines = dedented.splitlines()
        if not lines:
            return ""
        if not indent or len(lines) == 1:
            return dedented
        return lines[0] + "\n" + "\n".join(
            (indent + ln) if ln.strip() else ln
            for ln in lines[1:]
        )
