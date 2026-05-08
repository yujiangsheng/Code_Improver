"""Tests for ada_config + safe_run + make_plan."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.tools import TOOL_SCHEMAS, Tools
from ada.workspace import Workspace


class TestAdaConfig(unittest.TestCase):
    def test_reports_tool_count_and_env(self) -> None:
        with TemporaryDirectory() as td:
            tools = Tools(Workspace(Path(td)))
            r = tools.ada_config()
            self.assertEqual(r["tool_count"], len(TOOL_SCHEMAS))
            self.assertIn("env", r)
            self.assertIn("ADA_AUDIT", r["env"])
            self.assertIn("read_cache_size", r)


class TestSafeRun(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        os.environ.pop("ADA_GUARD_SECRETS", None)
        (self.root / "x.py").write_text("v=1\n")
        self.tools = Tools(Workspace(self.root))

    def test_success_keeps_changes(self) -> None:
        r = self.tools.safe_run("write_file", {"path": "y.py", "content": "v=2\n"})
        self.assertTrue(r["ok"])
        self.assertFalse(r["rolled_back"])
        self.assertTrue((self.root / "y.py").exists())

    def test_rollback_on_inner_error(self) -> None:
        # edit_file with missing string returns an error via exception → safe_run rollback.
        # Mutate the file first via write so we can verify rollback.
        (self.root / "x.py").write_text("ORIGINAL\n")
        # Re-create tools to ensure the safe_run sees latest mtime.
        tools = Tools(Workspace(self.root))
        r = tools.safe_run("edit_file", {"path": "x.py", "old": "NOT_THERE", "new": "Z"})
        self.assertFalse(r["ok"])
        self.assertTrue(r["rolled_back"])
        # File should be untouched.
        self.assertEqual((self.root / "x.py").read_text(), "ORIGINAL\n")

    def test_unknown_tool(self) -> None:
        r = self.tools.safe_run("does_not_exist", {})
        self.assertIn("error", r)


class TestMakePlanNoPlanner(unittest.TestCase):
    def test_returns_error_without_planner(self) -> None:
        with TemporaryDirectory() as td:
            tools = Tools(Workspace(Path(td)))
            # Default Tools has no planner attached.
            r = tools.make_plan("do something")
            self.assertIn("error", r)


class TestMakePlanWithFakePlanner(unittest.TestCase):
    def test_writes_artifact(self) -> None:
        with TemporaryDirectory() as td:
            tools = Tools(Workspace(Path(td)))
            tools._planner = lambda p: "## Goal\n- do x\n"
            r = tools.make_plan("ship feature")
            self.assertIn("plan", r)
            self.assertEqual(r["saved_to"], ".ada/plan.md")
            self.assertTrue((Path(td) / ".ada" / "plan.md").is_file())


if __name__ == "__main__":
    unittest.main()
