"""Smoke tests for the embedding index plumbing.

We use the ``ADA_EMBED_FAKE`` deterministic embedder so the tests stay
offline and reproducible.  The fake embedder cannot validate recall
quality — only storage, retrieval, and incremental-update behaviour.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.embed import EmbedIndex, _FakeEmbedder, _line_chunks


class TestLineChunks(unittest.TestCase):
    """The line-window splitter is the universal fallback chunker."""

    def test_short_file_one_chunk(self) -> None:
        out = _line_chunks("a\nb\nc\n")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 1)
        self.assertEqual(out[0][1], 3)

    def test_empty_input_no_chunks(self) -> None:
        self.assertEqual(_line_chunks(""), [])

    def test_overlap(self) -> None:
        text = "\n".join(f"line{i}" for i in range(1, 101))
        chunks = _line_chunks(text)
        # Each chunk covers 40 lines with 30-line step → ~4 chunks for 100 lines.
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0], 1)
        # Ranges must be increasing and contiguous-ish.
        for a, b in zip(chunks, chunks[1:]):
            self.assertGreater(b[0], a[0])


class TestFakeEmbedder(unittest.TestCase):
    """The fake embedder must be deterministic and unit-norm."""

    def test_deterministic(self) -> None:
        e = _FakeEmbedder(dim=64)
        a = e.embed(["hello world"])[0]
        b = e.embed(["hello world"])[0]
        self.assertEqual(a, b)

    def test_unit_norm(self) -> None:
        e = _FakeEmbedder(dim=128)
        v = e.embed(["anything"])[0]
        norm_sq = sum(x * x for x in v)
        self.assertAlmostEqual(norm_sq, 1.0, places=4)


class TestEmbedIndex(unittest.TestCase):
    """End-to-end index/search lifecycle with the fake embedder."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "a.py").write_text("def add(x, y):\n    return x + y\n")
        (self.root / "b.py").write_text(
            "def divide(x, y):\n    return x / y\n"
        )
        # Force the offline embedder regardless of caller env.
        os.environ["ADA_EMBED_FAKE"] = "1"
        self.idx = EmbedIndex(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        os.environ.pop("ADA_EMBED_FAKE", None)

    def test_available_with_fake(self) -> None:
        self.assertTrue(self.idx.available())

    def test_index_then_skip_unchanged(self) -> None:
        first = self.idx.index()
        self.assertEqual(first["indexed"], 2)
        self.assertGreater(first["chunks_added"], 0)
        # Re-running without changes must skip everything.
        second = self.idx.index()
        self.assertEqual(second["indexed"], 0)
        self.assertEqual(second["skipped"], 2)

    def test_search_returns_k_hits(self) -> None:
        self.idx.index()
        hits = self.idx.search("add two numbers", k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn(hits[0]["path"], {"a.py", "b.py"})
        self.assertIn("snippet", hits[0])
        self.assertIn("score", hits[0])

    def test_path_glob_filters(self) -> None:
        self.idx.index()
        only_a = self.idx.search("anything", k=5, path_glob="a.py")
        self.assertTrue(all(h["path"] == "a.py" for h in only_a))

    def test_deleted_file_is_pruned(self) -> None:
        self.idx.index()
        before = self.idx.stats()["files"]
        (self.root / "a.py").unlink()
        result = self.idx.index()
        self.assertGreaterEqual(result["deleted"], 1)
        self.assertEqual(self.idx.stats()["files"], before - 1)

    def test_clear(self) -> None:
        self.idx.index()
        self.idx.clear()
        self.assertEqual(self.idx.stats()["chunks"], 0)
        self.assertEqual(self.idx.stats()["files"], 0)


class TestUnavailableEmbedder(unittest.TestCase):
    """Without an API key (or ADA_EMBED_FAKE) the index reports unavailable."""

    def test_no_backend_no_results(self) -> None:
        # Snapshot env so we can restore.
        saved = {k: os.environ.pop(k, None) for k in (
            "ADA_EMBED_FAKE", "OPENAI_API_KEY", "ADA_EMBED_API_KEY",
        )}
        try:
            with TemporaryDirectory() as d:
                idx = EmbedIndex(d)
                self.assertFalse(idx.available())
                # search returns empty list rather than raising.
                self.assertEqual(idx.search("anything"), [])
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
