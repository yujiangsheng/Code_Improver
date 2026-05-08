"""End-to-end task runner for Ada.

Loads task fixtures from ``evals/tasks/<name>/``, copies each into a fresh
temporary git repo, runs Ada non-interactively against the goal, then
executes the task's ``verify.sh`` to determine pass / fail.

Each task directory must contain:

* ``task.json`` — ``{"goal": str, "max_steps": int, "max_cost": float?,
                     "verify_cmd": str, "timeout_sec": int?}``
* ``verify.sh`` — shell script returning exit code 0 = pass.
* Source files for the project under test (everything else in the dir).

Quick start
-----------
::

    # Dry run — list and parse tasks, no LLM calls:
    python -m evals.run_evals --list

    # Run all tasks against the configured Ada profile:
    python -m evals.run_evals

    # Run one task:
    python -m evals.run_evals --task fix_bug_stats

    # Persist results JSON for later analysis:
    python -m evals.run_evals --report results.json

Non-interactive mode
--------------------
Stdin is closed before launching Ada so any ``ask_user`` call returns ""
(the loop continues without human input).  Set ``ADA_PROFILE`` /
``ADA_WORKER_MODEL`` etc. in the calling environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"


@dataclass
class TaskSpec:
    """Parsed contents of one ``task.json`` file."""

    name: str
    path: Path
    goal: str
    verify_cmd: str
    max_steps: int = 25
    max_cost: float | None = None
    timeout_sec: int = 600
    description: str = ""

    @classmethod
    def load(cls, task_dir: Path) -> "TaskSpec":
        """Read ``task.json`` from *task_dir* and validate required fields."""
        spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        return cls(
            name=task_dir.name,
            path=task_dir,
            goal=spec["goal"],
            verify_cmd=spec["verify_cmd"],
            max_steps=int(spec.get("max_steps", 25)),
            max_cost=spec.get("max_cost"),
            timeout_sec=int(spec.get("timeout_sec", 600)),
            description=spec.get("description", ""),
        )


@dataclass
class TaskResult:
    """Outcome of one task run."""

    name: str
    passed: bool
    duration_sec: float
    verify_exit: int
    verify_output: str
    ada_exit: int
    ada_output_tail: str
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ─── discovery / preparation ────────────────────────────────────────────────────────────────────────────────


def discover_tasks(only: list[str] | None = None) -> list[TaskSpec]:
    """Return every task spec under ``evals/tasks/`` (optionally filtered)."""
    if not TASKS_DIR.is_dir():
        return []
    out: list[TaskSpec] = []
    for entry in sorted(TASKS_DIR.iterdir()):
        if not entry.is_dir() or not (entry / "task.json").is_file():
            continue
        if only and entry.name not in only:
            continue
        try:
            out.append(TaskSpec.load(entry))
        except (KeyError, json.JSONDecodeError) as exc:
            print(f"[warn] skipping {entry.name}: {exc}", file=sys.stderr)
    return out


def _prepare_workspace(task: TaskSpec, sandbox: Path) -> Path:
    """Copy task source files into *sandbox* and ``git init`` a baseline.

    The verify script and ``task.json`` are excluded from the copy so the
    agent doesn't see (or accidentally satisfy) the verifier directly.
    """
    target = sandbox / task.name
    target.mkdir(parents=True, exist_ok=True)
    for child in task.path.iterdir():
        if child.name in ("task.json", "verify.sh"):
            continue
        dest = target / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)

    # Ada wants a git repo (auto-branch creation, git_status, ...).
    env = {**os.environ, "GIT_AUTHOR_NAME": "evals", "GIT_AUTHOR_EMAIL": "evals@ada",
           "GIT_COMMITTER_NAME": "evals", "GIT_COMMITTER_EMAIL": "evals@ada"}
    subprocess.run(["git", "init", "-q"], cwd=target, env=env, check=False)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=target, env=env, check=False)
    subprocess.run(["git", "add", "-A"], cwd=target, env=env, check=False)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=target, env=env, check=False,
    )
    return target


# ─── execution ──────────────────────────────────────────────────────────────────────────────────────────────


def _run_ada(task: TaskSpec, project_dir: Path) -> tuple[int, str]:
    """Invoke ``main.py`` non-interactively against *project_dir*.

    Returns ``(exit_code, captured_stdout_stderr)``.  Stdin is closed so
    any ``ask_user`` calls auto-return "".
    """
    repo_root = ROOT.parent
    cmd = [
        sys.executable, str(repo_root / "main.py"),
        str(project_dir),
        "--goal", task.goal,
        "--steps", str(task.max_steps),
        "--no-auto-branch",  # we already created a branch
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=task.timeout_sec,
            cwd=str(repo_root),
        )
        return proc.returncode, proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        return 124, out + f"\n[evals] Ada exceeded {task.timeout_sec}s timeout\n"


def _run_verify(task: TaskSpec, project_dir: Path) -> tuple[int, str]:
    """Execute ``verify.sh`` against the (possibly modified) project."""
    verify_path = task.path / "verify.sh"
    if verify_path.is_file():
        proc = subprocess.run(
            ["bash", str(verify_path)],
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    else:
        # Fallback: run the inline verify_cmd in shell.
        proc = subprocess.run(
            task.verify_cmd, shell=True, cwd=str(project_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
        )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def run_task(task: TaskSpec, sandbox: Path) -> TaskResult:
    """Execute one task end-to-end and report the result."""
    t0 = time.monotonic()
    try:
        project_dir = _prepare_workspace(task, sandbox)
    except Exception as exc:
        return TaskResult(task.name, False, 0.0, -1, "", -1, "",
                          error=f"prepare failed: {exc}")
    ada_exit, ada_out = _run_ada(task, project_dir)
    verify_exit, verify_out = _run_verify(task, project_dir)
    return TaskResult(
        name=task.name,
        passed=verify_exit == 0,
        duration_sec=round(time.monotonic() - t0, 2),
        verify_exit=verify_exit,
        verify_output=verify_out[-2000:],
        ada_exit=ada_exit,
        ada_output_tail=ada_out[-2000:],
    )


# ─── CLI ────────────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point — see module docstring for usage examples."""
    parser = argparse.ArgumentParser(
        prog="ada-evals",
        description="Run end-to-end task evaluations for Ada.",
    )
    parser.add_argument("--task", action="append", default=None,
                        help="Run only the named task (repeatable).")
    parser.add_argument("--list", action="store_true",
                        help="List discovered tasks and exit (no LLM calls).")
    parser.add_argument("--report", default=None,
                        help="Write JSON results to this path.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the sandbox tempdir after running (for debugging).")
    args = parser.parse_args(argv)

    tasks = discover_tasks(args.task)
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        return 1
    if args.list:
        for t in tasks:
            cost = f", max_cost=${t.max_cost}" if t.max_cost else ""
            print(f"- {t.name} (max_steps={t.max_steps}{cost}): {t.goal[:80]}")
        return 0

    sandbox = Path(tempfile.mkdtemp(prefix="ada-evals-"))
    print(f"[evals] sandbox: {sandbox}")
    results: list[TaskResult] = []
    try:
        for t in tasks:
            print(f"[evals] running {t.name} ...")
            r = run_task(t, sandbox)
            mark = "PASS" if r.passed else "FAIL"
            print(f"[evals]   {mark}  {t.name}  ({r.duration_sec}s)")
            results.append(r)
    finally:
        if not args.keep:
            shutil.rmtree(sandbox, ignore_errors=True)
        else:
            print(f"[evals] sandbox kept at {sandbox}")

    passed = sum(1 for r in results if r.passed)
    print(f"\n[evals] {passed} / {len(results)} tasks passed")

    if args.report:
        Path(args.report).write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[evals] report written to {args.report}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
