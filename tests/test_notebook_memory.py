"""Tests for notebook helpers and persistent memory store."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.memstore import MemoryStore
from ada.notebook import edit_cell_source, read_notebook


def _make_nb(path: Path) -> None:
    nb = {
        "cells": [
            {"cell_type": "code", "source": ["print('hi')\n"], "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["hi\n"]}
            ], "execution_count": 1, "metadata": {}},
            {"cell_type": "markdown", "source": "# Title", "metadata": {}},
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


class TestNotebook(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.nb = Path(self._tmp.name) / "demo.ipynb"
        _make_nb(self.nb)

    def test_read(self) -> None:
        r = read_notebook(self.nb)
        self.assertEqual(r["cell_count"], 2)
        self.assertEqual(r["cells"][0]["type"], "code")
        self.assertIn("print", r["cells"][0]["source"])
        self.assertIn("hi", r["cells"][0]["outputs_brief"])
        self.assertEqual(r["kernel"], "python3")

    def test_edit_cell(self) -> None:
        r = edit_cell_source(self.nb, 0, "print('bye')\n")
        self.assertEqual(r["index"], 0)
        again = read_notebook(self.nb)
        self.assertIn("bye", again["cells"][0]["source"])
        # Outputs preserved.
        self.assertIn("hi", again["cells"][0]["outputs_brief"])

    def test_out_of_range(self) -> None:
        r = edit_cell_source(self.nb, 99, "x")
        self.assertIn("error", r)

    def test_missing_file(self) -> None:
        r = read_notebook(Path(self._tmp.name) / "nope.ipynb")
        self.assertIn("error", r)


class TestMemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = MemoryStore(Path(self._tmp.name) / "m.sqlite")

    def test_set_get_delete(self) -> None:
        self.store.set("ns1", "k", "v")
        r = self.store.get("ns1", "k")
        self.assertEqual(r["value"], "v")
        self.store.set("ns1", "k", "v2")
        self.assertEqual(self.store.get("ns1", "k")["value"], "v2")
        d = self.store.delete("ns1", "k")
        self.assertEqual(d["deleted"], 1)
        self.assertTrue(self.store.get("ns1", "k").get("missing"))

    def test_list_filtered(self) -> None:
        self.store.set("a", "k1", "1")
        self.store.set("b", "k2", "2")
        all_ = self.store.list()
        self.assertEqual(len(all_["entries"]), 2)
        only_a = self.store.list("a")
        self.assertEqual(len(only_a["entries"]), 1)
        self.assertEqual(only_a["entries"][0]["ns"], "a")

    def test_persist_across_instances(self) -> None:
        self.store.set("p", "k", "vv")
        again = MemoryStore(self.store.path)
        self.assertEqual(again.get("p", "k")["value"], "vv")


if __name__ == "__main__":
    unittest.main()
