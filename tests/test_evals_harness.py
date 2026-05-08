"""Smoke tests for the eval harness.

These tests don't actually invoke the LLM (which would require an API key
and burn cost).  They validate task discovery, spec parsing, sandbox
preparation, and the verify-only path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.run_evals import (
    TASKS_DIR,
    TaskSpec,
    _prepare_workspace,
    _run_verify,
    discover_tasks,
)


class TestDiscovery(unittest.TestCase):
    """Every shipped task must have a valid task.json + verify.sh."""

    def test_at_least_five_tasks(self) -> None:
        tasks = discover_tasks()
        self.assertGreaterEqual(len(tasks), 5)

    def test_filter_by_name(self) -> None:
        tasks = discover_tasks(["fix_bug_stats"])
        self.assertEqual([t.name for t in tasks], ["fix_bug_stats"])

    def test_each_task_has_verify_script(self) -> None:
        for t in discover_tasks():
            self.assertTrue(
                (t.path / "verify.sh").is_file(),
                f"{t.name} missing verify.sh",
            )

    def test_specs_have_required_fields(self) -> None:
        for t in discover_tasks():
            self.assertTrue(t.goal.strip(), f"{t.name} has empty goal")
            self.assertTrue(t.verify_cmd.strip(), f"{t.name} has empty verify_cmd")
            self.assertGreater(t.max_steps, 0)


class TestWorkspacePrep(unittest.TestCase):
    """Sandbox preparation must copy sources and init a git repo."""

    def test_prepare_excludes_meta_files(self) -> None:
        task = TaskSpec.load(TASKS_DIR / "fix_bug_stats")
        with TemporaryDirectory() as d:
            project = _prepare_workspace(task, Path(d))
            # Source files copied:
            self.assertTrue((project / "stats.py").is_file())
            self.assertTrue((project / "test_stats.py").is_file())
            # Meta files NOT copied — Ada must not see verify.sh or task.json:
            self.assertFalse((project / "verify.sh").exists())
            self.assertFalse((project / "task.json").exists())
            # Git repo created with baseline commit:
            self.assertTrue((project / ".git").is_dir())
            log = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=project, stdout=subprocess.PIPE, check=True,
            ).stdout.decode()
            self.assertIn("baseline", log)


class TestVerifyOnly(unittest.TestCase):
    """The verify path can be exercised without running Ada."""

    def test_verify_passes_after_manual_fix(self) -> None:
        # Simulate a successful agent run by manually fixing the bug, then
        # check that verify.sh reports pass.  This exercises _run_verify
        # without invoking the LLM.
        task = TaskSpec.load(TASKS_DIR / "fix_bug_stats")
        with TemporaryDirectory() as d:
            project = _prepare_workspace(task, Path(d))
            stats = project / "stats.py"
            text = stats.read_text(encoding="utf-8")
            stats.write_text(text.replace("(len(xs) - 1)", "len(xs)"), encoding="utf-8")
            exit_code, output = _run_verify(task, project)
            self.assertEqual(exit_code, 0, msg=output[-500:])

    def test_verify_fails_on_unmodified_baseline(self) -> None:
        task = TaskSpec.load(TASKS_DIR / "fix_bug_stats")
        with TemporaryDirectory() as d:
            project = _prepare_workspace(task, Path(d))
            exit_code, _ = _run_verify(task, project)
            self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
