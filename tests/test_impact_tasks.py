"""Tests for impact analysis, diff stats, and task queue."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.impact import changed_test_targets, diff_stats
from ada.tasks import TaskQueue


class TestImpact(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("")
        (self.root / "pkg" / "core.py").write_text("def f():\n    return 1\n")
        (self.root / "pkg" / "other.py").write_text("def g():\n    return 2\n")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_core.py").write_text(
            "from pkg.core import f\n\ndef test_f():\n    assert f() == 1\n"
        )
        (self.root / "tests" / "test_other.py").write_text(
            "from pkg.other import g\n\ndef test_g():\n    assert g() == 2\n"
        )

    def test_only_affected_tests_returned(self) -> None:
        r = changed_test_targets(self.root, ["pkg/core.py"])
        self.assertIn("tests/test_core.py", r["tests"])
        self.assertNotIn("tests/test_other.py", r["tests"])

    def test_no_python_changes(self) -> None:
        r = changed_test_targets(self.root, ["README.md"])
        self.assertEqual(r["tests"], [])
        self.assertEqual(r["modules"], [])


class TestDiffStats(unittest.TestCase):
    def test_safe_verdict(self) -> None:
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,3 @@\n"
            " keep\n"
            "+added line\n"
            "-removed line\n"
        )
        r = diff_stats(diff)
        self.assertEqual(r["verdict"], "safe")
        self.assertEqual(r["files"][0]["path"], "foo.py")
        self.assertEqual(r["files"][0]["added"], 1)
        self.assertEqual(r["files"][0]["removed"], 1)

    def test_risky_when_huge(self) -> None:
        body = "+x\n" * 250
        diff = "--- a/big.py\n+++ b/big.py\n@@\n" + body
        r = diff_stats(diff)
        self.assertEqual(r["verdict"], "risky")


class TestTaskQueue(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = TaskQueue(Path(self._tmp.name) / "tasks.json")

    def test_add_and_list(self) -> None:
        a = self.q.add("first")
        b = self.q.add("second")
        items = self.q.list()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], a["id"])
        self.assertEqual(items[1]["title"], "second")
        self.assertEqual(b["status"], "pending")

    def test_update_and_remove(self) -> None:
        a = self.q.add("first")
        r = self.q.update(a["id"], "done")
        self.assertEqual(r["status"], "done")
        bad = self.q.update(a["id"], "wat")
        self.assertIn("error", bad)
        rm = self.q.remove(a["id"])
        self.assertTrue(rm.get("ok"))
        self.assertEqual(self.q.list(), [])

    def test_persist_across_instances(self) -> None:
        self.q.add("one")
        q2 = TaskQueue(self.q.path)
        self.assertEqual(len(q2.list()), 1)


if __name__ == "__main__":
    unittest.main()
