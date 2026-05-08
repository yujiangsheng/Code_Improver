"""Trajectory replay: pretty-print a saved Ada audit log.

The audit log written by :mod:`ada.audit` is JSONL, one line per tool call.
This module turns it back into a readable timeline for post-mortems::

    python -m ada.replay .ada/audit.jsonl
    python -m ada.replay .ada/audit.jsonl --summary
    python -m ada.replay .ada/audit.jsonl --tool run_tests

We deliberately stay read-only: replay never re-executes tools (no risk of
side-effects on disk).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audit import load, summarise

console = Console()


def render_timeline(entries: list[dict], tool_filter: str | None = None) -> None:
    """Print one row per tool call: step, tool, ok, ms, args/result preview."""
    table = Table(title="Ada tool-call timeline", show_lines=False)
    table.add_column("step", justify="right", style="cyan")
    table.add_column("tool", style="bold")
    table.add_column("ok", justify="center")
    table.add_column("ms", justify="right", style="dim")
    table.add_column("args", overflow="fold")
    table.add_column("result", overflow="fold")
    for e in entries:
        if tool_filter and e.get("tool") != tool_filter:
            continue
        ok_str = "[green]✓[/green]" if e.get("ok") else "[red]✗[/red]"
        table.add_row(
            str(e.get("step", "?")),
            str(e.get("tool", "?")),
            ok_str,
            str(e.get("duration_ms", 0)),
            str(e.get("args_brief", ""))[:80],
            str(e.get("result_brief", ""))[:120],
        )
    console.print(table)


def render_summary(entries: list[dict]) -> None:
    """Aggregate stats across the run (calls, errors, hot tools)."""
    s = summarise(entries)
    console.print(Panel(
        f"calls: {s['total_calls']}    errors: {s['errors']}    "
        f"total_ms: {s['total_ms']}",
        title="Run summary",
        border_style="cyan",
    ))
    table = Table(title="Time per tool")
    table.add_column("tool", style="bold")
    table.add_column("calls", justify="right")
    table.add_column("errors", justify="right", style="red")
    table.add_column("ms", justify="right")
    for tool, stats in s["by_tool"].items():
        table.add_row(
            tool,
            str(stats["calls"]),
            str(stats["errors"]),
            str(stats["ms"]),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m ada.replay``."""
    parser = argparse.ArgumentParser(description="Replay an Ada audit log")
    parser.add_argument("path", help="path to audit.jsonl")
    parser.add_argument("--summary", action="store_true",
                        help="aggregate stats only (no timeline)")
    parser.add_argument("--tool", default=None,
                        help="filter timeline to one tool name")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        console.print(f"[red]not found: {path}[/red]")
        return 2
    entries = load(path)
    if not entries:
        console.print("[yellow]audit log is empty[/yellow]")
        return 0
    if args.summary:
        render_summary(entries)
    else:
        render_timeline(entries, tool_filter=args.tool)
        render_summary(entries)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main(sys.argv[1:]))
