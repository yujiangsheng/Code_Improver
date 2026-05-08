"""Tests for traceback parsing + locate_failures + batch_edit + secret guard."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.tools import Tools
from ada.traceback_parser import Frame, in_workspace, parse
from ada.workspace import Workspace


SAMPLE_TB = """\
============================= test session starts ==============================
collected 1 item
tests/test_thing.py F                                                    [100%]
=================================== FAILURES ===================================
___________________________ test_does_a_thing _________________________________
Traceback (most recent call last):
  File "/repo/tests/test_thing.py", line 14, in test_does_a_thing
    self.assertEqual(do_thing(2), 4)
  File "/repo/src/lib.py", line 7, in do_thing
    return x * x - 1
AssertionError: 3 != 4
"""


class TestTracebackParser(unittest.TestCase):
    def test_parse_frames_dedup(self) -> None:
        frames = parse(SAMPLE_TB)
        # Two unique File-style frames.
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(frames[0].symbol, "test_does_a_thing")
        self.assertEqual(frames[1].symbol, "do_thing")

    def test_in_workspace_filters(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "lib.py").write_text("def do_thing(x):\n    return x*x-1\n")
            tb = SAMPLE_TB.replace("/repo/", str(root) + "/")
            (root / "tests").mkdir()
            (root / "tests" / "test_thing.py").write_text("x = 1\n" * 20)
            frames = in_workspace(parse(tb), root)
            self.assertTrue(any(f.file == "src/lib.py" for f in frames))


class TestLocateFailures(unittest.TestCase):
    def test_locate_failures_returns_preview(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "lib.py").write_text(
                "def do_thing(x):\n" * 5 + "    return x*x-1\n" + "x=1\n" * 10
            )
            tb = f'  File "{root}/src/lib.py", line 6, in do_thing\n    return x*x-1\n'
            tools = Tools(Workspace(root))
            r = tools.locate_failures(tb)
            self.assertEqual(r["count"], 1)
            self.assertEqual(r["frames"][0]["file"], "src/lib.py")
            self.assertIn("return", r["frames"][0]["preview"])


class TestBatchEdit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "a.py").write_text("FOO = 1\n")
        (self.root / "b.py").write_text("BAR = 2\n")
        os.environ.pop("ADA_GUARD_SECRETS", None)
        self.tools = Tools(Workspace(self.root))

    def test_apply_all(self) -> None:
        r = self.tools.batch_edit([
            {"path": "a.py", "old": "FOO = 1", "new": "FOO = 10"},
            {"path": "b.py", "old": "BAR = 2", "new": "BAR = 20"},
        ])
        self.assertEqual(r["count"], 2)
        self.assertEqual((self.root / "a.py").read_text(), "FOO = 10\n")
        self.assertEqual((self.root / "b.py").read_text(), "BAR = 20\n")

    def test_rollback_on_failure(self) -> None:
        # Second edit will fail (string not found); first must be rolled back.
        r = self.tools.batch_edit([
            {"path": "a.py", "old": "FOO = 1", "new": "FOO = 99"},
            {"path": "b.py", "old": "NOT_THERE", "new": "x"},
        ])
        self.assertIn("error", r)
        self.assertEqual((self.root / "a.py").read_text(), "FOO = 1\n")
        self.assertEqual((self.root / "b.py").read_text(), "BAR = 2\n")


class TestSecretGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        os.environ.pop("ADA_GUARD_SECRETS", None)
        self.tools = Tools(Workspace(self.root))

    def test_write_file_blocks_aws_key(self) -> None:
        # AKIA + 16 chars is the AWS access-key pattern in safety.py.
        body = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
        r = self.tools.write_file("creds.py", body)
        self.assertIn("error", r)
        self.assertFalse((self.root / "creds.py").exists())

    def test_allow_secrets_bypass(self) -> None:
        body = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
        r = self.tools.write_file("creds.py", body, allow_secrets=True)
        self.assertIn("path", r)
        self.assertTrue((self.root / "creds.py").exists())

    def test_env_off_disables(self) -> None:
        os.environ["ADA_GUARD_SECRETS"] = "0"
        try:
            tools = Tools(Workspace(self.root))
            body = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
            r = tools.write_file("creds.py", body)
            self.assertIn("path", r)
        finally:
            os.environ.pop("ADA_GUARD_SECRETS", None)


if __name__ == "__main__":
    unittest.main()
