"""Auto-detect & run a project's test / lint / typecheck commands.

Closes one of the largest gaps Ada has against Claude Code & Aider:
*structural editing without verification feedback*. The agent now has two
high-level tools — :py:meth:`run_tests` and :py:meth:`run_lint` — that
detect the project's tooling once, run the relevant commands, and return a
trimmed structured result the LLM can act on.

Detection is purely heuristic (presence of marker files / config sections).
Anything not detected falls back to a no-op with a friendly message rather
than blowing up.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ── command catalogue ────────────────────────────────────────────────────
#
# Each entry: (display name, marker-fn, command list to run sequentially).
# Markers are evaluated against the project root.  The first matching
# entry wins for ``which``-style detection — but for ``run_*`` we run
# *every* matching toolchain so polyglot repos (Python + JS + Go in one
# tree) all get covered.

@dataclass(frozen=True)
class _Cmd:
    name: str           # e.g. "pytest"
    detect: tuple[str, ...]   # any of these files/dirs in root → enabled
    cmd: str            # shell command
    parse_summary: str = ""   # regex or literal hint shown in result


_TEST_CMDS: tuple[_Cmd, ...] = (
    _Cmd(
        "pytest",
        ("pytest.ini", "pyproject.toml", "setup.cfg", "tests/", "tests"),
        "python -m pytest -q --maxfail=5 --no-header",
    ),
    _Cmd(
        "unittest",
        ("test/", "tests/"),
        "python -m unittest discover -q",
    ),
    _Cmd(
        "go-test",
        ("go.mod",),
        "go test ./...",
    ),
    _Cmd(
        "cargo-test",
        ("Cargo.toml",),
        "cargo test --quiet",
    ),
    _Cmd(
        "npm-test",
        ("package.json",),
        "npm test --silent",
    ),
)


_LINT_CMDS: tuple[_Cmd, ...] = (
    _Cmd("ruff",         ("pyproject.toml", "ruff.toml", ".ruff.toml"),
         "python -m ruff check ."),
    _Cmd("mypy",         ("mypy.ini", "pyproject.toml"),
         "python -m mypy --no-error-summary --pretty ."),
    _Cmd("eslint",       (".eslintrc", ".eslintrc.js", ".eslintrc.json",
                          "eslint.config.js", "eslint.config.mjs"),
         "npx --no-install eslint . --max-warnings=0"),
    _Cmd("tsc",          ("tsconfig.json",),
         "npx --no-install tsc --noEmit"),
    _Cmd("go-vet",       ("go.mod",),
         "go vet ./..."),
    _Cmd("clippy",       ("Cargo.toml",),
         "cargo clippy --quiet -- -D warnings"),
)


_FMT_CMDS: tuple[_Cmd, ...] = (
    _Cmd("black",        ("pyproject.toml",), "python -m black --check ."),
    _Cmd("prettier",     ("package.json", ".prettierrc"),
         "npx --no-install prettier --check ."),
    _Cmd("gofmt",        ("go.mod",), "gofmt -l ."),
)


# ── small helpers ────────────────────────────────────────────────────────


def _has_marker(root: Path, markers: tuple[str, ...]) -> bool:
    for m in markers:
        p = root / m
        if p.exists():
            return True
    return False


def _toml_section_present(root: Path, dotted: str) -> bool:
    """Cheap pyproject.toml section probe (no toml dep needed)."""
    py = root / "pyproject.toml"
    if not py.is_file():
        return False
    try:
        text = py.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return f"[{dotted}]" in text or f"[tool.{dotted}]" in text


def _has_pytest(root: Path) -> bool:
    return (
        (root / "pytest.ini").is_file()
        or (root / "tests").is_dir()
        or (root / "test").is_dir()
        or _toml_section_present(root, "pytest.ini_options")
        or _toml_section_present(root, "pytest")
    )


def _has_ruff(root: Path) -> bool:
    return (
        (root / "ruff.toml").is_file()
        or (root / ".ruff.toml").is_file()
        or _toml_section_present(root, "ruff")
        or _toml_section_present(root, "ruff.lint")
    )


def _has_mypy(root: Path) -> bool:
    return (
        (root / "mypy.ini").is_file()
        or _toml_section_present(root, "mypy")
    )


def _has_black(root: Path) -> bool:
    return _toml_section_present(root, "black")


# Per-tool refined predicates (override the marker-only check above).
_PRED = {
    "pytest": _has_pytest,
    "ruff":   _has_ruff,
    "mypy":   _has_mypy,
    "black":  _has_black,
}


def _detected(cmd: _Cmd, root: Path) -> bool:
    pred = _PRED.get(cmd.name)
    if pred is not None:
        return pred(root)
    return _has_marker(root, cmd.detect)


def _truncate_tail(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[-cap:]


# ── public engine ────────────────────────────────────────────────────────


class Verifier:
    """Detect & execute test / lint / format commands for a project root."""

    def __init__(self, root: Path, default_timeout: int = 180) -> None:
        self.root = Path(root)
        self.default_timeout = default_timeout

    # ── detection ──

    def detect(self) -> dict:
        """Return the toolchains we'd run for tests, lint, and format."""
        return {
            "tests": [c.name for c in _TEST_CMDS if _detected(c, self.root)],
            "lint":  [c.name for c in _LINT_CMDS if _detected(c, self.root)],
            "format": [c.name for c in _FMT_CMDS if _detected(c, self.root)],
        }

    # ── execution ──

    def _run_one(self, cmd: _Cmd, timeout: int) -> dict:
        try:
            proc = subprocess.run(
                cmd.cmd,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            return {
                "tool": cmd.name,
                "cmd": cmd.cmd,
                "exit_code": proc.returncode,
                "ok": proc.returncode == 0,
                "stdout": _truncate_tail(stdout, 4000),
                "stderr": _truncate_tail(stderr, 2000),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "tool": cmd.name,
                "cmd": cmd.cmd,
                "exit_code": -1,
                "ok": False,
                "stdout": _truncate_tail(exc.stdout or "", 4000),
                "stderr": f"[timeout after {exc.timeout}s]",
            }
        except FileNotFoundError as exc:
            return {
                "tool": cmd.name,
                "cmd": cmd.cmd,
                "exit_code": -2,
                "ok": False,
                "stdout": "",
                "stderr": f"executable not found: {exc}",
            }

    def run_tests(
        self,
        only: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        return self._run_set(_TEST_CMDS, "tests", only, timeout)

    def run_lint(
        self,
        only: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        return self._run_set(_LINT_CMDS, "lint", only, timeout)

    def run_format_check(
        self,
        only: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        return self._run_set(_FMT_CMDS, "format", only, timeout)

    def _run_set(
        self,
        catalogue: tuple[_Cmd, ...],
        kind: str,
        only: list[str] | None,
        timeout: int | None,
    ) -> dict:
        wanted = set(only) if only else None
        runs = []
        for c in catalogue:
            if wanted is not None and c.name not in wanted:
                continue
            if not _detected(c, self.root):
                continue
            runs.append(self._run_one(c, timeout or self.default_timeout))
        if not runs:
            return {
                "kind": kind,
                "ran": [],
                "ok": True,
                "note": (
                    f"no {kind} toolchain detected in this project "
                    "(or none of the requested tools matched)"
                ),
            }
        all_ok = all(r["ok"] for r in runs)
        return {
            "kind": kind,
            "ran": [r["tool"] for r in runs],
            "ok": all_ok,
            "results": runs,
        }
