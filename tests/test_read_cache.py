"""Tests for the read-file cache embedded in ada.tools.Tools.read_file."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.tools import Tools
from ada.workspace import Workspace


class TestReadFileCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "a.txt").write_text("hello\nworld\nfoo\n")
        ws = Workspace(self.root)
        self.tools = Tools(ws)

    def test_first_read_is_uncached(self) -> None:
        r = self.tools.read_file("a.txt", 1, 10)
        self.assertNotIn("cached", r)
        self.assertEqual(self.tools._read_cache_hits, 0)

    def test_repeat_read_is_cached(self) -> None:
        self.tools.read_file("a.txt", 1, 10)
        r2 = self.tools.read_file("a.txt", 1, 10)
        self.assertTrue(r2.get("cached"))
        self.assertIn("hint", r2)
        self.assertEqual(self.tools._read_cache_hits, 1)

    def test_different_window_misses(self) -> None:
        self.tools.read_file("a.txt", 1, 10)
        r2 = self.tools.read_file("a.txt", 2, 3)
        self.assertNotIn("cached", r2)

    def test_mtime_change_invalidates(self) -> None:
        self.tools.read_file("a.txt", 1, 10)
        # Modify content so size+mtime change → cache key differs.
        import time as _t
        _t.sleep(0.01)
        (self.root / "a.txt").write_text("brand new contents here\n")
        r2 = self.tools.read_file("a.txt", 1, 10)
        self.assertNotIn("cached", r2)
        self.assertIn("brand new", r2["content"])


if __name__ == "__main__":
    unittest.main()
