"""Unit tests for ada/verify.py and ada/conventions.py."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ada.conventions import detected_files, load_conventions
from ada.verify import Verifier


class TestVerifierDetect(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_repo_detects_nothing(self) -> None:
        d = Verifier(self.root).detect()
        self.assertEqual(d["tests"], [])
        self.assertEqual(d["lint"], [])

    def test_pytest_marker_directory(self) -> None:
        (self.root / "tests").mkdir()
        d = Verifier(self.root).detect()
        self.assertIn("pytest", d["tests"])

    def test_pyproject_with_ruff_section_detected(self) -> None:
        (self.root / "pyproject.toml").write_text(
            "[tool.ruff]\nline-length = 100\n"
        )
        d = Verifier(self.root).detect()
        self.assertIn("ruff", d["lint"])
        # No mypy section -> mypy must NOT be flagged.
        self.assertNotIn("mypy", d["lint"])

    def test_go_mod_marker(self) -> None:
        (self.root / "go.mod").write_text("module x\n")
        d = Verifier(self.root).detect()
        self.assertIn("go-test", d["tests"])
        self.assertIn("go-vet", d["lint"])

    def test_run_returns_noop_when_nothing_detected(self) -> None:
        v = Verifier(self.root)
        self.assertTrue(v.run_tests()["ok"])
        self.assertTrue(v.run_lint()["ok"])
        self.assertIn("note", v.run_tests())


class TestConventions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_no_files_returns_empty(self) -> None:
        self.assertEqual(load_conventions(self.root), "")
        self.assertEqual(detected_files(self.root), [])

    def test_agents_md_loaded(self) -> None:
        (self.root / "AGENTS.md").write_text("rules!")
        out = load_conventions(self.root)
        self.assertIn("AGENTS.md", out)
        self.assertIn("rules!", out)

    def test_multiple_files_concatenated(self) -> None:
        (self.root / "AGENTS.md").write_text("agent rules")
        (self.root / "CLAUDE.md").write_text("claude rules")
        out = load_conventions(self.root)
        self.assertIn("AGENTS.md", out)
        self.assertIn("CLAUDE.md", out)

    def test_truncation(self) -> None:
        (self.root / "AGENTS.md").write_text("x" * 20000)
        out = load_conventions(self.root)
        self.assertIn("[truncated]", out)


if __name__ == "__main__":
    unittest.main()
