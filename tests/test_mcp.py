"""Smoke tests for the MCP (Model Context Protocol) stdio client.

We spawn a tiny in-process Python script as a fake MCP server so the
tests stay self-contained — no Node, no npx, no network.
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ada.mcp import MCPClient, MCPError, MCPServer

# Tiny MCP-protocol-compliant server: implements initialize, tools/list,
# tools/call (echo).  Written as a string so each test gets a fresh subprocess.
FAKE_SERVER = textwrap.dedent("""
    import json, sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "echo"},
                "capabilities": {"tools": {}},
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
                {"name": "echo",
                 "description": "Echo a string back",
                 "inputSchema": {"type": "object",
                                  "properties": {"text": {"type": "string"}},
                                  "required": ["text"]}}
            ]}})
        elif method == "tools/call":
            args = msg["params"]["arguments"]
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": "echoed: " + args["text"]}]
            }})
        else:
            send({"jsonrpc": "2.0", "id": msg.get("id"),
                   "error": {"code": -32601, "message": "method not found"}})
""")

# A separate server that crashes on initialize so we can test error capture.
BROKEN_SERVER = "import sys; sys.exit(1)"


def _write_server(dirpath: Path, body: str) -> Path:
    """Drop *body* into a Python file and make it importable."""
    p = dirpath / "srv.py"
    p.write_text(body, encoding="utf-8")
    return p


class TestMCPHandshake(unittest.TestCase):
    """Initialize → tools/list must populate the registry."""

    def test_handshake_and_tool_call(self) -> None:
        with TemporaryDirectory() as d:
            dp = Path(d)
            srv_path = _write_server(dp, FAKE_SERVER)
            srv = MCPServer("echo", sys.executable, [str(srv_path)])
            try:
                self.assertTrue(srv.initialize(), msg=srv.last_error)
                self.assertEqual(len(srv.tools), 1)
                self.assertEqual(srv.tools[0]["name"], "echo")
                resp = srv.call_tool("echo", {"text": "hello"})
                # Per MCP, the response is wrapped in {result: {content: [...]}}
                self.assertIn("result", resp)
                self.assertEqual(
                    resp["result"]["content"][0]["text"], "echoed: hello"
                )
            finally:
                srv.stop()


class TestMCPClient(unittest.TestCase):
    """The aggregator must namespace tools and route calls correctly."""

    def test_load_from_config(self) -> None:
        with TemporaryDirectory() as d:
            dp = Path(d)
            srv_path = _write_server(dp, FAKE_SERVER)
            cfg_path = dp / "mcp.json"
            cfg_path.write_text(json.dumps({
                "mcpServers": {
                    "echo": {"command": sys.executable, "args": [str(srv_path)]}
                }
            }))
            cli = MCPClient.from_config(cfg_path)
            try:
                self.assertEqual(list(cli.servers), ["echo"])
                self.assertEqual(cli.errors, [])
                schemas = cli.tool_schemas()
                names = [s["function"]["name"] for s in schemas]
                self.assertIn("mcp_echo_echo", names)
                self.assertTrue(cli.has("mcp_echo_echo"))

                # Tool call returns a flattened {text, content, ...} dict.
                result = cli.call("mcp_echo_echo", {"text": "world"})
                self.assertEqual(result["text"], "echoed: world")
            finally:
                cli.stop_all()

    def test_missing_config_is_noop(self) -> None:
        with TemporaryDirectory() as d:
            cli = MCPClient.from_config(Path(d) / "nope.json")
            self.assertEqual(cli.servers, {})
            self.assertEqual(cli.errors, [])

    def test_disabled_server_skipped(self) -> None:
        with TemporaryDirectory() as d:
            dp = Path(d)
            cfg = dp / "mcp.json"
            cfg.write_text(json.dumps({
                "mcpServers": {
                    "off": {"command": "anything", "enabled": False}
                }
            }))
            cli = MCPClient.from_config(cfg)
            self.assertEqual(cli.servers, {})

    def test_broken_server_recorded_in_errors(self) -> None:
        with TemporaryDirectory() as d:
            dp = Path(d)
            srv_path = _write_server(dp, BROKEN_SERVER)
            cfg = dp / "mcp.json"
            cfg.write_text(json.dumps({
                "mcpServers": {
                    "broken": {"command": sys.executable, "args": [str(srv_path)]}
                }
            }))
            cli = MCPClient.from_config(cfg)
            try:
                self.assertEqual(cli.servers, {})
                self.assertEqual(len(cli.errors), 1)
                self.assertIn("broken", cli.errors[0])
            finally:
                cli.stop_all()

    def test_unknown_tool_returns_error(self) -> None:
        cli = MCPClient()
        result = cli.call("mcp_nope_nope", {})
        self.assertIn("error", result)


class TestNameSafety(unittest.TestCase):
    """OpenAI requires tool names to match ``^[a-zA-Z0-9_-]+$``."""

    def test_special_chars_replaced(self) -> None:
        # Underscore replacement keeps the name OpenAI-compliant.
        name = MCPClient._ada_tool_name("file.system", "read/file")
        self.assertEqual(name, "mcp_file_system_read_file")

    def test_truncation(self) -> None:
        long = "x" * 200
        name = MCPClient._ada_tool_name(long, long)
        self.assertLessEqual(len(name), 64)


if __name__ == "__main__":
    unittest.main()
