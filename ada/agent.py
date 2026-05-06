"""The Ada agent — tool-using loop driven by an LLM and the prompt.md spec."""
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

console = Console()


def _load_prompt() -> str:
    here = Path(__file__).resolve().parent.parent
    return (here / "prompt.md").read_text(encoding="utf-8")


class Ada:
    def __init__(self, target_dir: str, user_goal: str = "",
                 model: str | None = None,
                 planner_model: str | None = None,
                 max_steps: int | None = None,
                 cmd_timeout: int | None = None,
                 auto_branch: bool = True):
        self.ws = Workspace(target_dir)
        self.llm = LLM(model=model)
        self.planner = Planner(model=planner_model)
        self.tools = Tools(
            self.ws,
            cmd_timeout=cmd_timeout or int(os.getenv("ADA_CMD_TIMEOUT", "120")),
            planner=self.planner.advise,
        )
        self.max_steps = max_steps or int(os.getenv("ADA_MAX_STEPS", "80"))
        self.user_goal = user_goal.strip() or "(not provided — please ask the user)"

        # Auto-create a working branch so the user can diff / revert easily.
        branch_note = self._maybe_create_branch(auto_branch)

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
                    "Begin by executing your Kickoff Protocol (Section 8 of the prompt). "
                    "Then enter The Ada Loop. Use the provided tools — do not write code "
                    "in plain text answers. After each verified Backlog item, call "
                    "`git_commit` with a concise message. For hard planning calls, use "
                    "`consult_planner`. When you need human input, call `ask_user`. "
                    "When fully done, call `finish`."
                ),
            },
        ]
        self._done_summary: str | None = None

    def _maybe_create_branch(self, auto_branch: bool) -> str:
        git = Git(self.ws.target_dir)
        if not git.is_repo():
            return "target is NOT a git repo — git_* tools will be unavailable"
        if not auto_branch:
            return f"on branch {git.current_branch()} (auto-branch disabled)"
        from datetime import datetime
        name = f"ada/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            git.create_branch(name)
            console.print(f"[green]✓ created working branch: {name}[/green]")
            return f"working on new branch '{name}' (auto-created)"
        except GitError as e:
            console.print(f"[yellow]could not create branch: {e}[/yellow]")
            return f"on branch {git.current_branch()} (auto-branch failed: {e})"

    # ---- main loop ---------------------------------------------------------
    def run(self) -> str:
        for step in range(1, self.max_steps + 1):
            console.rule(f"[bold cyan]Step {step}")
            try:
                resp = self.llm.chat(self.messages, TOOL_SCHEMAS)
            except Exception as e:
                console.print(f"[red]LLM error: {e}")
                return f"LLM error: {e}"

            msg = resp.choices[0].message
            # echo any free-form content
            if msg.content:
                console.print(Panel(msg.content, title="Ada", border_style="cyan"))

            # serialize assistant message back into history
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

            if not msg.tool_calls:
                # No tool call & no finish → nudge back into the loop
                self.messages.append({
                    "role": "user",
                    "content": "Reminder: you must drive progress via tool calls "
                               "(or call `finish` to stop). Continue the Ada Loop.",
                })
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch(name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                })
                if self._done_summary is not None:
                    console.rule("[bold green]Ada finished")
                    console.print(Panel(self._done_summary, title="Final Summary",
                                        border_style="green"))
                    return self._done_summary

        msg = f"Reached max_steps={self.max_steps} without finishing."
        console.print(f"[yellow]{msg}")
        return msg

    # ---- tool dispatch -----------------------------------------------------
    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        console.print(f"[dim]→ tool: {name}({_brief(args)})[/dim]")
        try:
            if name == "ask_user":
                answer = _ask(args.get("question", ""))
                return json.dumps({"user_reply": answer}, ensure_ascii=False)
            if name == "finish":
                self._done_summary = args.get("summary", "(no summary)")
                return json.dumps({"acknowledged": True})

            fn = getattr(self.tools, name, None)
            if fn is None:
                return json.dumps({"error": f"unknown tool {name!r}"})
            result = fn(**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _ask(question: str) -> str:
    console.print(Panel(question, title="Ada asks you", border_style="yellow"))
    try:
        return input(">>> ").strip()
    except EOFError:
        return ""
