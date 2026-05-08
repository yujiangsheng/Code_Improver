"""Tests for checkpoint store + cross-session recall + preview_diff."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.checkpoint import CheckpointStore
from ada.recall import build_recall
from ada.tools import Tools
from ada.workspace import Workspace


class TestCheckpointStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "a.py").write_text("v=1\n")
        (self.root / "b.py").write_text("v=2\n")
        self.store = CheckpointStore(self.root)

    def test_create_and_restore(self) -> None:
        cp = self.store.create(label="before")
        self.assertGreater(len(cp.files), 0)
        # Mutate.
        (self.root / "a.py").write_text("MUTATED\n")
        (self.root / "b.py").write_text("MUTATED2\n")
        r = self.store.restore(cp.id)
        self.assertEqual(r["restored"], len(cp.files))
        self.assertEqual((self.root / "a.py").read_text(), "v=1\n")
        self.assertEqual((self.root / "b.py").read_text(), "v=2\n")

    def test_list_newest_first(self) -> None:
        cp1 = self.store.create(label="one")
        import time as _t
        _t.sleep(1.1)  # ids are second-resolution
        cp2 = self.store.create(label="two")
        listing = self.store.list()
        ids = [c["id"] for c in listing]
        self.assertEqual(ids.index(cp2.id), 0)
        self.assertEqual(ids.index(cp1.id), 1)

    def test_restore_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.restore("does-not-exist")

    def test_skip_dirs(self) -> None:
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("nope")
        cp = self.store.create()
        for f in cp.files:
            self.assertFalse(f.startswith(".git"))


class TestRecall(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_empty_when_no_state(self) -> None:
        self.assertEqual(build_recall(self.root), "")

    def test_picks_up_journal_and_audit(self) -> None:
        ada_dir = self.root / ".ada"
        ada_dir.mkdir()
        (ada_dir / "journal.md").write_text("entry one\nentry two\n")
        (ada_dir / "backlog.md").write_text("- [ ] do thing\n")
        (ada_dir / "audit.jsonl").write_text(
            json.dumps({"step": 1, "tool": "read_file", "ok": True, "duration_ms": 5}) + "\n"
        )
        blob = build_recall(self.root)
        self.assertIn("recent journal", blob)
        self.assertIn("entry two", blob)
        self.assertIn("backlog", blob)
        self.assertIn("read_file", blob)


class TestPreviewDiffNoRepo(unittest.TestCase):
    def test_returns_error_outside_git(self) -> None:
        with TemporaryDirectory() as td:
            tools = Tools(Workspace(Path(td)))
            r = tools.preview_diff()
            self.assertIn("error", r)


class TestCheckpointTools(unittest.TestCase):
    def test_create_and_list_via_tools(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "x.py").write_text("hi\n")
            tools = Tools(Workspace(root))
            r = tools.create_checkpoint(label="t1")
            self.assertIn("id", r)
            listing = tools.list_checkpoints()
            self.assertEqual(len(listing["checkpoints"]), 1)


if __name__ == "__main__":
    unittest.main()
