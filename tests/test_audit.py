"""Tests for ada.audit (JSONL audit log) and ada.replay (post-mortem CLI)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.audit import AuditLog, load, summarise
from ada.replay import main as replay_main


class TestAuditLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "nested" / "audit.jsonl"
        # Force enable regardless of env.
        os.environ["ADA_AUDIT"] = "1"
        self.log = AuditLog(self.path)

    def test_record_writes_jsonl(self) -> None:
        self.log.record(1, "read_file", '{"path":"x"}', '{"ok":true}', True, 12)
        self.log.record(2, "run_tests", "{}", '{"failed":1}', False, 250, error="AssertionError")
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        a = json.loads(lines[0])
        self.assertEqual(a["tool"], "read_file")
        self.assertTrue(a["ok"])
        self.assertEqual(a["duration_ms"], 12)
        b = json.loads(lines[1])
        self.assertFalse(b["ok"])
        self.assertEqual(b["error"], "AssertionError")

    def test_disable_via_env(self) -> None:
        os.environ["ADA_AUDIT"] = "0"
        log = AuditLog(self.path.parent / "off.jsonl")
        log.record(1, "x", "{}", "{}", True, 1)
        self.assertFalse((self.path.parent / "off.jsonl").exists())

    def test_load_skips_malformed(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"step": 1, "tool": "a", "ok": True, "duration_ms": 5}) + "\n"
            + "garbage\n"
            + json.dumps({"step": 2, "tool": "b", "ok": False, "duration_ms": 7}) + "\n"
        )
        rows = load(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["tool"], "a")
        self.assertEqual(rows[1]["tool"], "b")

    def test_summarise_aggregates(self) -> None:
        entries = [
            {"step": 1, "tool": "read_file", "ok": True, "duration_ms": 10},
            {"step": 2, "tool": "read_file", "ok": True, "duration_ms": 30},
            {"step": 3, "tool": "run_tests", "ok": False, "duration_ms": 200},
        ]
        s = summarise(entries)
        self.assertEqual(s["total_calls"], 3)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["total_ms"], 240)
        # Sorted by ms desc → run_tests first.
        self.assertEqual(next(iter(s["by_tool"])), "run_tests")
        self.assertEqual(s["by_tool"]["read_file"]["calls"], 2)


class TestReplayCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "audit.jsonl"

    def test_missing_file_returns_2(self) -> None:
        self.assertEqual(replay_main([str(self.path)]), 2)

    def test_empty_file_returns_0(self) -> None:
        self.path.write_text("")
        self.assertEqual(replay_main([str(self.path)]), 0)

    def test_full_run_returns_0(self) -> None:
        self.path.write_text(
            json.dumps({"step": 1, "tool": "read_file", "ok": True, "duration_ms": 5,
                        "args_brief": "{}", "result_brief": "{}"}) + "\n"
        )
        self.assertEqual(replay_main([str(self.path), "--summary"]), 0)
        self.assertEqual(replay_main([str(self.path), "--tool", "read_file"]), 0)


if __name__ == "__main__":
    unittest.main()
