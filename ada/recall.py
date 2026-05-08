"""Cross-session memory recall: brief context from prior runs.

At Ada startup we surface a short summary of:

* the last N journal entries (free-form notes the agent left for itself),
* the current backlog (what the agent thought was next),
* the last K tool calls from the audit log (so the agent sees what was
  in flight when the previous run ended).

The output is a single Markdown blob suitable for prepending to the
system prompt.  Empty when no prior state exists, so first runs are
unaffected.
"""
from __future__ import annotations

from pathlib import Path

from .audit import load as load_audit


def build_recall(
    ws_root: Path,
    journal_lines: int = 40,
    audit_entries: int = 8,
    max_chars: int = 2000,
) -> str:
    """Return a Markdown recall blob, or "" when there's nothing to show."""
    parts: list[str] = []
    ws_root = Path(ws_root)

    journal = ws_root / ".ada" / "journal.md"
    if journal.is_file():
        text = journal.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-journal_lines:]).strip()
        if tail:
            parts.append("### recent journal\n" + tail)

    backlog = ws_root / ".ada" / "backlog.md"
    if backlog.is_file():
        text = backlog.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            parts.append("### backlog\n" + text)

    audit = ws_root / ".ada" / "audit.jsonl"
    if audit.is_file():
        entries = load_audit(audit)[-audit_entries:]
        if entries:
            lines = [
                f"- step {e.get('step')}: {e.get('tool')} "
                f"({'ok' if e.get('ok') else 'ERR'}, {e.get('duration_ms')}ms)"
                for e in entries
            ]
            parts.append("### last tool calls (previous run)\n" + "\n".join(lines))

    if not parts:
        return ""

    blob = "## prior-session recall\n\n" + "\n\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n... [truncated]"
    return blob
