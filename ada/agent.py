"""The Ada agent \u2014 orchestrates the LLM + tool-calling loop.

Architecture
------------
Ada runs a ``for step in range(max_steps)`` loop:

  1. Send the current ``messages`` list to the worker LLM.
  2. If the model returns tool calls, dispatch each one via ``_dispatch()``.
  3. Append both the assistant turn and the tool-result turns to ``messages``
     so the model has full history on the next iteration.
  4. Repeat until the model calls ``finish`` or ``max_steps`` is reached.

Special tool handling
---------------------
``ask_user`` and ``finish`` are intercepted by ``_dispatch()`` before reaching
``Tools``:

* ``ask_user``  \u2014 prints the question and blocks on stdin (CLI) or an asyncio
  event (web server).  The web server replaces ``_dispatch`` with a patched
  version that uses ``SessionState.wait_for_feedback()`` instead.
* ``finish``    \u2014 stores the summary and sets ``_done_summary``, which the loop
  checks after every dispatch to break early.

Nudge on empty tool calls
--------------------------
If the model generates a text-only response (no tool calls) without calling
``finish``, Ada appends a reminder message so the model gets back on track
rather than silently halting.

Console output
--------------
All output goes through the module-level ``console`` (a Rich ``Console``).
The web server monkey-patches this to a ``QueueConsole`` that also pushes
every ``print`` call into the SSE event queue, giving the browser real-time
log streaming.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .git_ops import Git, GitError
from .llm import LLM, Planner
from .tools import TOOL_SCHEMAS, Tools
from .workspace import Workspace

# Module-level console \u2014 replaced by a QueueConsole in web server mode.
console = Console()


def _load_prompt() -> str:
    """Load ``prompt.md`` from the project root (sibling of the ``ada/`` package)."""
    here = Path(__file__).resolve().parent.parent
    return (here / "prompt.md").read_text(encoding="utf-8")


class Ada:
    """Autonomous code-improvement agent.

    Parameters
    ----------
    target_dir:
        Path to the project to improve.  Must exist as a directory.
    user_goal:
        Plain-text goal, e.g. ``"\u8ba9\u6240\u6709\u6d4b\u8bd5\u901a\u8fc7\u5e76\u6d88\u9664 lint \u8b66\u544a"``.
    model:
        Worker model name.  Overrides ``ADA_WORKER_MODEL`` env var.
    planner_model:
        Planner model name.  Overrides ``ADA_PLANNER_MODEL`` env var.
    max_steps:
        Hard cap on tool-loop iterations.  Overrides ``ADA_MAX_STEPS`` env var.
    cmd_timeout:
        Per-shell-command timeout in seconds.  Overrides ``ADA_CMD_TIMEOUT``.
    auto_branch:
        If ``True`` (default), create an ``ada/<YYYYMMDD-HHmmSS>`` git branch
        before the first step so the main branch stays clean.
    """

    def __init__(
        self,
        target_dir: str,
        user_goal: str = "",
        model: str | None = None,
        planner_model: str | None = None,
        max_steps: int | None = None,
        cmd_timeout: int | None = None,
        auto_branch: bool = True,
    ) -> None:
        self.ws = Workspace(target_dir)
        self.llm = LLM(model=model)
        self.planner = Planner(model=planner_model)
        self.tools = Tools(
            self.ws,
            cmd_timeout=cmd_timeout or int(os.getenv("ADA_CMD_TIMEOUT", "120")),
            planner=self.planner.advise,
        )
        self.max_steps = max_steps or int(os.getenv("ADA_MAX_STEPS", "80"))
        self.user_goal = user_goal.strip() or "(not provided \u2014 please ask the user)"

        # Auto-create a working branch; note for the initial user message.
        branch_note = self._maybe_create_branch(auto_branch)

        # Build the initial conversation: system prompt + first user message.
        # The user message tells Ada where to start and what constraints apply.
        self.messages: list[dict] = [
            {"role": "system", "content": _load_prompt()},
            {
                "role": "user",
                "content": (
                    f"TARGET_DIR: {self.ws.target_dir}\n"
                    f"USER_GOAL: {self.user_goal}\n"
                    f"WORKER_MODEL: {self.llm.model}\n"
                    f"PLANNER_MODEL: {self.planner.model}\n"
                    f"GIT: {branch_note}\n\n"
                    "Begin by executing your Kickoff Protocol (Section\u00a08 of the prompt). "
                    "Then enter The Ada Loop.  Use the provided tools \u2014 do not write code "
                    "in plain-text answers.  After each verified Backlog item, call "
                    "`git_commit` with a concise message.  For hard planning questions, use "
                    "`consult_planner`.  When you need human input, call `ask_user`. "
                    "When fully done, call `finish`."
                ),
            },
        ]
        # Set by _dispatch when the model calls `finish`.
        self._done_summary: str | None = None

    # \u2500\u2500 setup helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _maybe_create_branch(self, auto_branch: bool) -> str:
        """Attempt to create a dedicated working branch; return a status note."""
        git = Git(self.ws.target_dir)
        if not git.is_repo():
            return "target is NOT a git repo \u2014 git_* tools will be unavailable"
        if not auto_branch:
            return f"on branch {git.current_branch()} (auto-branch disabled)"
        from datetime import datetime
        name = f"ada/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            git.create_branch(name)
            console.print(f"[green]\u2713 created working branch: {name}[/green]")
            return f"working on new branch '{name}' (auto-created)"
        except GitError as exc:
            console.print(f"[yellow]could not create branch: {exc}[/yellow]")
            return f"on branch {git.current_branch()} (auto-branch failed: {exc})"

    # \u2500\u2500 main loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def run(self) -> str:
        """Run the tool-calling loop until ``finish`` or ``max_steps``.

        Returns
        -------
        str
            The final summary string (from ``finish``) or a timeout message.
        """
        for step in range(1, self.max_steps + 1):
            console.rule(f"[bold cyan]Step {step}")

            # \u2500 call LLM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            try:
                resp = self.llm.chat(self.messages, TOOL_SCHEMAS)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}")
                return f"LLM error: {exc}"

            msg = resp.choices[0].message

            # Echo any free-form reasoning/narration the model produces.
            if msg.content:
                console.print(Panel(msg.content, title="Ada", border_style="cyan"))

            # Serialise the assistant turn back into conversation history.
            # The ``tool_calls`` list must be included verbatim so the next
            # turn can reference tool call IDs correctly.
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ] or None,
            })

            # \u2500 no tool calls \u2014 nudge the model back on track \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            if not msg.tool_calls:
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Reminder: you must drive progress via tool calls "
                        "(or call `finish` to stop).  Continue the Ada Loop."
                    ),
                })
                continue

            # \u2500 dispatch each tool call \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}  # malformed JSON \u2014 pass empty dict and let dispatch handle it

                result = self._dispatch(name, args)

                # Add the tool result to history so the model sees it next turn.
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                })

                # Check if `finish` was just called.
                if self._done_summary is not None:
                    console.rule("[bold green]Ada finished")
                    console.print(
                        Panel(self._done_summary, title="Final Summary", border_style="green")
                    )
                    return self._done_summary

        # Reached the step cap without the model calling finish.
        msg_out = f"Reached max_steps={self.max_steps} without finishing."
        console.print(f"[yellow]{msg_out}")
        return msg_out

    # \u2500\u2500 tool dispatch \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool by *name* with *args*, returning a JSON string.

        ``ask_user`` and ``finish`` are handled here rather than delegated to
        ``Tools``, because they affect control flow (blocking I/O and loop
        termination respectively).

        All exceptions are caught and returned as ``{"error": "..."}`` JSON so
        the LLM can self-correct without the Python process crashing.
        """
        console.print(f"[dim]\u2192 tool: {name}({_brief(args)})[/dim]")
        try:
            if name == "ask_user":
                # Block until the human provides input.
                answer = _ask(args.get("question", ""))
                return json.dumps({"user_reply": answer}, ensure_ascii=False)

            if name == "finish":
                # Signal the run loop to stop after this dispatch.
                self._done_summary = args.get("summary", "(no summary)")
                return json.dumps({"acknowledged": True})

            fn = getattr(self.tools, name, None)
            if fn is None:
                return json.dumps({"error": f"unknown tool {name!r}"})
            result = fn(**args)
            return json.dumps(result, ensure_ascii=False, default=str)

        except Exception as exc:
            return json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
            )


# \u2500\u2500 helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _brief(args: dict) -> str:
    """Format tool arguments as a compact single-line string for the console."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:60] + '...' if len(s) > 60 else s}")
    return ", ".join(parts)


def _ask(question: str) -> str:
    """Print *question* and block until the user types a reply (CLI mode)."""
    console.print(Panel(question, title="Ada asks you", border_style="yellow"))
    try:
        return input(">>> ").strip()
    except EOFError:
        return ""  # non-interactive environment (pipe, test, etc.)
