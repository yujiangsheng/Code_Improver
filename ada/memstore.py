"""SQLite-backed key/value memory store for Ada.

Survives across runs (lives at ``.ada/memory.sqlite``). Designed for
small, frequently-recalled facts: model preferences, recurring
gotchas, cached lookup results.

Schema is intentionally tiny — one table::

    CREATE TABLE memory (
        ns      TEXT NOT NULL,
        key     TEXT NOT NULL,
        value   TEXT NOT NULL,
        updated REAL NOT NULL,
        PRIMARY KEY (ns, key)
    )

Per workspace (and per process) we use a thread-local connection to
sidestep the macOS Python 3.9 + sqlite3 segfault on shared connections.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class MemoryStore:
    """Tiny namespace-keyed K/V on top of SQLite (WAL-mode, thread-local)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        # Bootstrap schema in the calling thread; cache that connection
        # in _tls so we don't leak an orphan on first use. Other threads
        # initialise lazily via the property below (each gets its own
        # _tls slot — threading.local() guarantees no cross-thread races).
        boot = self._connect()
        boot.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory (
                ns      TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL,
                updated REAL NOT NULL,
                PRIMARY KEY (ns, key)
            );
            """
        )
        self._tls.conn = boot

    # ── connection management ──────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        """Return this thread's connection (lazily created)."""
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = self._connect()
            self._tls.conn = c
        return c

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection with WAL + busy_timeout for safety."""
        c = sqlite3.connect(str(self.path), isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    # ── CRUD ───────────────────────────────────────────────────────────

    def set(self, ns: str, key: str, value: str) -> dict:
        """Upsert ``(ns, key) -> value``."""
        self.conn.execute(
            "INSERT OR REPLACE INTO memory(ns, key, value, updated) VALUES (?,?,?,?)",
            (ns, key, value, time.time()),
        )
        return {"ns": ns, "key": key, "bytes": len(value)}

    def get(self, ns: str, key: str) -> dict:
        """Fetch one value or return ``{"missing": True}``."""
        row = self.conn.execute(
            "SELECT value, updated FROM memory WHERE ns=? AND key=?", (ns, key)
        ).fetchone()
        if row is None:
            return {"ns": ns, "key": key, "missing": True}
        return {"ns": ns, "key": key, "value": row[0], "updated": row[1]}

    def list(self, ns: str | None = None) -> dict:
        """List entries, optionally filtered to a single namespace."""
        if ns is None:
            rows = self.conn.execute(
                "SELECT ns, key, length(value), updated FROM memory ORDER BY updated DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT ns, key, length(value), updated FROM memory "
                "WHERE ns=? ORDER BY updated DESC",
                (ns,),
            ).fetchall()
        return {
            "entries": [
                {"ns": r[0], "key": r[1], "bytes": r[2], "updated": r[3]} for r in rows
            ],
        }

    def delete(self, ns: str, key: str) -> dict:
        """Remove one entry; idempotent."""
        cur = self.conn.execute(
            "DELETE FROM memory WHERE ns=? AND key=?", (ns, key)
        )
        return {"ns": ns, "key": key, "deleted": cur.rowcount}
