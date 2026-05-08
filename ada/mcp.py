"""MCP (Model Context Protocol) stdio client.

Connects Ada to any MCP server (filesystem, GitHub, Linear, Postgres, ...)
that speaks the standard stdio transport.  Each server's tools are
auto-discovered via ``tools/list`` and exposed as Ada tools with the
namespaced name ``mcp_<server>_<tool>``.

Lifecycle
---------
1. ``MCPClient.from_config(path)`` reads ``.ada/mcp.json``.
2. For each entry it spawns the server as a subprocess (stdin/stdout pipes)
   and runs the JSON-RPC handshake:
      → ``initialize`` (capability negotiation)
      ← server capabilities
      → ``notifications/initialized``
      → ``tools/list``
      ← tool catalogue
3. Tool calls are routed via :py:meth:`call`; responses block on a
   per-request ``threading.Event`` keyed by JSON-RPC id.
4. ``stop_all()`` terminates every subprocess on shutdown.

Config schema (``.ada/mcp.json``)
----------------------------------
::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/repo"],
          "env": {"NODE_ENV": "production"},
          "cwd": "/repo",
          "enabled": true
        }
      }
    }

Robustness
----------
Failures spawning or initialising a server are caught and logged into
``self.errors`` so a single broken entry doesn't disable the others.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# JSON-RPC request that times out if the server doesn't respond promptly.
_DEFAULT_RPC_TIMEOUT = float(os.getenv("ADA_MCP_TIMEOUT", "30"))
# How long to wait for the initial handshake before giving up on a server.
_INIT_TIMEOUT = float(os.getenv("ADA_MCP_INIT_TIMEOUT", "15"))
# Protocol version we advertise — matches the 2024-11-05 spec.
_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    """Raised for protocol- or transport-level failures."""


class MCPServer:
    """Single MCP subprocess + JSON-RPC reader thread."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.tools: list[dict[str, Any]] = []
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        # id → {"event": Event, "result": dict|None}
        self._pending: dict[int, dict[str, Any]] = {}
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._closed = False
        self.last_error: str | None = None

        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        try:
            self._proc = subprocess.Popen(
                [command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                cwd=cwd,
                bufsize=0,
            )
        except (OSError, FileNotFoundError) as exc:
            self.last_error = f"spawn failed: {exc}"
            return

        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{name}-reader", daemon=True
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._drain_stderr, name=f"mcp-{name}-stderr", daemon=True
        )
        self._stderr_reader.start()

    # ── lifecycle ───────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Run the MCP handshake and populate ``self.tools``.

        Returns ``True`` on success, ``False`` (with ``self.last_error``
        populated) on any failure.
        """
        if self._proc is None:
            return False
        try:
            init_resp = self.request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ada", "version": "1.0"},
                },
                timeout=_INIT_TIMEOUT,
            )
            if "error" in init_resp:
                self.last_error = f"initialize: {init_resp['error']}"
                return False
            self._notify("notifications/initialized", {})
            tools_resp = self.request("tools/list", {}, timeout=_INIT_TIMEOUT)
            if "error" in tools_resp:
                self.last_error = f"tools/list: {tools_resp['error']}"
                return False
            self.tools = (tools_resp.get("result") or {}).get("tools") or []
            return True
        except Exception as exc:
            self.last_error = f"initialize: {type(exc).__name__}: {exc}"
            return False

    def stop(self) -> None:
        """Best-effort terminate the subprocess and reader threads."""
        self._closed = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        # Wake any pending requests so callers don't deadlock.
        for entry in self._pending.values():
            entry["event"].set()

    # ── JSON-RPC ────────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_RPC_TIMEOUT,
    ) -> dict[str, Any]:
        """Send a request and block until the matching response arrives."""
        if self._proc is None or self._closed:
            raise MCPError(f"server {self.name!r} not running")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
        event = threading.Event()
        self._pending[rid] = {"event": event, "result": None}
        msg = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {},
        }
        self._send(msg)
        if not event.wait(timeout=timeout):
            self._pending.pop(rid, None)
            raise MCPError(f"timeout waiting for {method}")
        result = self._pending.pop(rid)["result"]
        return result or {"error": {"message": "server closed connection"}}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool by its server-side name."""
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, msg: dict[str, Any]) -> None:
        """Write one newline-delimited JSON message to the server's stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("stdin closed")
        line = (json.dumps(msg) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise MCPError(f"server pipe broken: {exc}") from exc

    def _read_loop(self) -> None:
        """Background thread: parse server stdout into responses."""
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            if not raw:
                break
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            rid = msg.get("id")
            if rid is None:
                continue  # notification or unsolicited message — ignore
            entry = self._pending.get(rid)
            if entry is not None:
                entry["result"] = msg
                entry["event"].set()
        # Stream closed — wake everyone waiting.
        for entry in self._pending.values():
            entry["event"].set()

    def _drain_stderr(self) -> None:
        """Consume stderr to prevent the subprocess from blocking on a full pipe."""
        assert self._proc is not None and self._proc.stderr is not None
        for _ in self._proc.stderr:
            pass


# ─── client ─────────────────────────────────────────────────────────────────────────────────────────────────


class MCPClient:
    """Aggregates multiple MCP servers and exposes their tools to Ada."""

    def __init__(self) -> None:
        self.servers: dict[str, MCPServer] = {}
        # Ada-namespaced name → (server name, original tool name).
        self.tool_map: dict[str, tuple[str, str]] = {}
        self.errors: list[str] = []

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MCPClient":
        """Build a client from an ``mcp.json`` file (no-op if missing)."""
        client = cls()
        p = Path(config_path)
        if not p.is_file():
            return client
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            client.errors.append(f"failed to read {p}: {exc}")
            return client

        servers = cfg.get("mcpServers") or cfg.get("servers") or {}
        for name, spec in servers.items():
            if not spec.get("enabled", True):
                continue
            try:
                client.add_server(
                    name=name,
                    command=spec["command"],
                    args=spec.get("args"),
                    env=spec.get("env"),
                    cwd=spec.get("cwd"),
                )
            except Exception as exc:
                client.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return client

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> MCPServer:
        """Spawn a server, run handshake, register its tools."""
        srv = MCPServer(name, command, args=args, env=env, cwd=cwd)
        if srv.last_error:
            self.errors.append(f"{name}: {srv.last_error}")
            return srv
        if not srv.initialize():
            self.errors.append(f"{name}: {srv.last_error or 'init failed'}")
            srv.stop()
            return srv
        self.servers[name] = srv
        for tool in srv.tools:
            ada_name = self._ada_tool_name(name, tool["name"])
            self.tool_map[ada_name] = (name, tool["name"])
        return srv

    # ── tool surface for the agent ──────────────────────────────────────

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Convert MCP tool defs into OpenAI function-calling schemas."""
        out: list[dict[str, Any]] = []
        for ada_name, (srv_name, tool_name) in self.tool_map.items():
            srv = self.servers[srv_name]
            tool = next((t for t in srv.tools if t["name"] == tool_name), None)
            if tool is None:
                continue
            params = tool.get("inputSchema") or {"type": "object", "properties": {}}
            # OpenAI is strict about JSON Schema dialect; ensure top-level type.
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object", "properties": {}}
            description = tool.get("description") or f"MCP tool from {srv_name}"
            out.append({
                "type": "function",
                "function": {
                    "name": ada_name,
                    "description": f"[mcp:{srv_name}] {description[:300]}",
                    "parameters": params,
                },
            })
        return out

    def has(self, ada_name: str) -> bool:
        """Return ``True`` if *ada_name* refers to a routed MCP tool."""
        return ada_name in self.tool_map

    def call(self, ada_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a namespaced MCP tool call; normalise the response."""
        if ada_name not in self.tool_map:
            return {"error": f"unknown MCP tool {ada_name!r}"}
        srv_name, tool_name = self.tool_map[ada_name]
        srv = self.servers.get(srv_name)
        if srv is None:
            return {"error": f"MCP server {srv_name!r} not running"}
        try:
            resp = srv.call_tool(tool_name, args)
        except MCPError as exc:
            return {"error": f"MCP {srv_name}: {exc}"}
        if "error" in resp and resp["error"]:
            return {"error": resp["error"]}
        return self._flatten_result(resp.get("result"))

    def stop_all(self) -> None:
        """Terminate every spawned subprocess."""
        for srv in self.servers.values():
            srv.stop()
        self.servers.clear()
        self.tool_map.clear()

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _ada_tool_name(server: str, tool: str) -> str:
        """Build a unique Ada tool name like ``mcp_filesystem_read_file``.

        Replaces non-alphanumeric chars with ``_`` to satisfy the OpenAI
        function-name regex (``^[a-zA-Z0-9_-]+$``).
        """
        safe = lambda s: "".join(c if c.isalnum() or c in "_-" else "_" for c in s)
        return f"mcp_{safe(server)}_{safe(tool)}"[:64]

    @staticmethod
    def _flatten_result(result: Any) -> dict[str, Any]:
        """Convert MCP ``content`` arrays to plain ``{"text", "data"}`` dicts.

        MCP returns ``{"content": [{"type": "text", "text": "..."}, ...]}``;
        callers (i.e. the LLM) just want the consolidated text.
        """
        if not isinstance(result, dict):
            return {"result": result}
        content = result.get("content")
        if not content:
            return result
        texts: list[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text") or "")
        flat = dict(result)
        flat["text"] = "\n".join(texts) if texts else None
        return flat
