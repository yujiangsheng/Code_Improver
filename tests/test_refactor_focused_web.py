"""Tests for rename_symbol + run_focused_tests + web_fetch."""
from __future__ import annotations

import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.refactor import rename_symbol
from ada.tools import Tools
from ada.web import fetch_url
from ada.workspace import Workspace


class TestRenameSymbol(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "a.py").write_text("def OldName():\n    return OldName.x\n")
        (self.root / "b.py").write_text("from a import OldName\n# unrelated: OldNameish\n")

    def test_dry_run_does_not_modify(self) -> None:
        r = rename_symbol(self.root, "OldName", "NewName", dry_run=True)
        self.assertEqual(r["files_changed"], 0)
        self.assertEqual(r["total_replacements"], 3)
        self.assertIn("OldName", (self.root / "a.py").read_text())

    def test_real_rename(self) -> None:
        r = rename_symbol(self.root, "OldName", "NewName")
        self.assertEqual(r["files_changed"], 2)
        # Word-boundary protected: OldNameish must remain.
        b = (self.root / "b.py").read_text()
        self.assertIn("OldNameish", b)
        self.assertIn("from a import NewName", b)
        a = (self.root / "a.py").read_text()
        self.assertIn("def NewName", a)
        self.assertIn("NewName.x", a)

    def test_invalid_identifier_rejected(self) -> None:
        r = rename_symbol(self.root, "Old Name", "X")
        self.assertIn("error", r)
        r2 = rename_symbol(self.root, "OldName", "OldName")
        self.assertIn("error", r2)


class TestRunFocusedTestsFallback(unittest.TestCase):
    def test_skips_when_no_repo(self) -> None:
        with TemporaryDirectory() as td:
            tools = Tools(Workspace(Path(td)))
            r = tools.run_focused_tests()
            self.assertTrue(r.get("skipped"))


# A tiny throwaway HTTP server for the web_fetch tests.
class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><head><title>T</title></head>"
                         b"<body><script>x=1</script><p>Hello world</p></body></html>")

    def log_message(self, *_args):  # silence
        return


class TestWebFetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_https_only_blocks_http_default(self) -> None:
        r = fetch_url(f"http://127.0.0.1:{self.port}/")
        self.assertIn("error", r)

    def test_http_with_override(self) -> None:
        import os
        os.environ["ADA_FETCH_ALLOW_HTTP"] = "1"
        try:
            r = fetch_url(f"http://127.0.0.1:{self.port}/")
        finally:
            os.environ.pop("ADA_FETCH_ALLOW_HTTP", None)
        self.assertEqual(r.get("status"), 200)
        self.assertEqual(r.get("title"), "T")
        self.assertIn("Hello world", r.get("text", ""))
        self.assertNotIn("x=1", r.get("text", ""))  # script stripped

    def test_bad_scheme(self) -> None:
        r = fetch_url("file:///etc/passwd")
        self.assertIn("error", r)

    def test_allowlist_blocks_other_hosts(self) -> None:
        import os
        os.environ["ADA_FETCH_ALLOWLIST"] = "example.com"
        os.environ["ADA_FETCH_ALLOW_HTTP"] = "1"
        try:
            r = fetch_url(f"http://127.0.0.1:{self.port}/")
        finally:
            os.environ.pop("ADA_FETCH_ALLOWLIST", None)
            os.environ.pop("ADA_FETCH_ALLOW_HTTP", None)
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()
