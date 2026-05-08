"""Tiny persistent task queue for multi-goal sessions.

Stores ordered TODO items in ``.ada/tasks.json`` with status ('pending',
'in_progress', 'done', 'blocked').  No dependencies between tasks (kept
deliberately simple); the agent owns prioritisation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_STATUSES = ("pending", "in_progress", "done", "blocked")


class TaskQueue:
    """Disk-backed list of ``{id, title, status, created, updated}`` items."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def list(self) -> list[dict]:
        """Return all tasks (oldest first)."""
        return self._load()

    def add(self, title: str) -> dict:
        """Append a new pending task; returns the created entry."""
        items = self._load()
        new_id = (max((t["id"] for t in items), default=0) + 1)
        entry = {
            "id": new_id,
            "title": title,
            "status": "pending",
            "created": time.time(),
            "updated": time.time(),
        }
        items.append(entry)
        self._save(items)
        return entry

    def update(self, task_id: int, status: str) -> dict:
        """Move *task_id* to *status* and stamp ``updated``."""
        if status not in _STATUSES:
            return {"error": f"status must be one of {_STATUSES}"}
        items = self._load()
        for t in items:
            if t["id"] == task_id:
                t["status"] = status
                t["updated"] = time.time()
                self._save(items)
                return t
        return {"error": f"no task with id={task_id}"}

    def remove(self, task_id: int) -> dict:
        """Drop a task from the queue."""
        items = self._load()
        new_items = [t for t in items if t["id"] != task_id]
        if len(new_items) == len(items):
            return {"error": f"no task with id={task_id}"}
        self._save(new_items)
        return {"ok": True, "removed": task_id}
