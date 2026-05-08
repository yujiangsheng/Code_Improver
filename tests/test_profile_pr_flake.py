"""Tests for profile_run, flake_check, generate_pr_description."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.profile import profile_python
from ada.tools import Tools
from ada.workspace import Workspace


class TestProfilePython(unittest.TestCase):
    def test_basic_snippet(self) -> None:
        r = profile_python("sum(range(1000))", top=5)
        self.assertIn("top", r)
        self.assertGreater(r["total_funcs"], 0)
        self.assertLessEqual(len(r["top"]), 5)

    def test_error_caught(self) -> None:
        r = profile_python("1/0")
        self.assertIn("error", r)
        self.assertIn("ZeroDivisionError", r["error"])


class TestFlakeCheck(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tools = Tools(Workspace(Path(self._tmp.name)))

    def test_runs_must_be_at_least_two(self) -> None:
        r = self.tools.flake_check(paths=["."], runs=1)
        self.assertIn("error", r)

    def test_stable_when_command_consistent(self) -> None:
        # `pytest` will fail because there are no tests in the temp
        # workspace, but the failure should be stable across runs.
        r = self.tools.flake_check(paths=["."], runs=2)
        self.assertIn("verdict", r)
        self.assertEqual(r["verdict"], "stable")
        self.assertEqual(len(r["runs"]), 2)


class TestGeneratePrDescription(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tools = Tools(Workspace(Path(self._tmp.name)))

    def test_no_repo(self) -> None:
        r = self.tools.generate_pr_description(ref="HEAD")
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()
