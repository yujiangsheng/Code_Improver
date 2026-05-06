"""Ada Web Server — Flask backend with SSE streaming and REST API.

Architecture overview
---------------------
Each browser session maps to a ``SessionState`` object that owns:

  * ``q``               — thread-safe queue of SSE payloads
  * ``feedback_event``  — threading.Event unblocked when user sends feedback
  * ``stop_requested``  — threading.Event set by the Stop button

Ada runs in a **daemon thread** per session.  The thread monkey-patches the
module-level ``ada.agent.console`` with a ``QueueConsole`` that pushes every
Rich ``print()`` call into ``state.q`` so the browser gets real-time output.

SSE event types
---------------
  step         — current loop iteration number
  assistant    — free-form LLM narration
  tool         — tool call (name + abbreviated args)
  tool_result  — tool return value
  log          — Rich console output
  ask_user     — Ada is paused waiting for human input
  user_replied — echo of the injected reply
  finish       — final summary; Ada is done
  error        — unhandled exception from the Ada thread
  done         — session ended (success or error)
  ping         — heartbeat every 20 s to keep the connection alive

REST endpoints
--------------
  GET  /                  → serve web/index.html
  POST /api/start         → start a new Ada session; returns {session_id}
  POST /api/feedback/<id> → unblock a waiting ask_user
  GET  /api/stream/<id>   → SSE event stream
  POST /api/stop/<id>     → request graceful stop
  GET  /api/status/<id>   → {running, done, step, summary}
  GET  /api/models        → list available Ollama models
  GET  /api/browse        → list subdirectories (for directory picker UI)
  POST /api/shutdown      → gracefully exit the Flask process
"""
from __future__ import annotations

import json
import os
import queue
import signal
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

load_dotenv()

# ── app setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "web" / "static"))

# ── session registry ─────────────────────────────────────────────────────────
# { session_id: SessionState }
_sessions: dict[str, "SessionState"] = {}
_sessions_lock = threading.Lock()


class SessionState:
    """All mutable state for a single Ada run.

    Lives in ``_sessions`` for the process lifetime; created by ``/api/start``
    and never explicitly deleted (memory is bounded by session count).
    """

    def __init__(
        self,
        sid: str,
        target_dir: str,
        goal: str,
        model: str,
        planner_model: str,
        max_steps: int,
        cmd_timeout: int,
    ) -> None:
        self.sid = sid
        self.target_dir = target_dir
        self.goal = goal
        self.model = model
        self.planner_model = planner_model
        self.max_steps = max_steps
        self.cmd_timeout = cmd_timeout

        # SSE output queue — capped so a slow browser can't fill RAM.
        self.q: queue.Queue[str] = queue.Queue(maxsize=2000)
        # Feedback channel: Ada blocks here until the user sends a reply.
        self.feedback_event = threading.Event()
        self.feedback_value: str = ""
        # Set by /api/stop to ask the Ada thread to exit gracefully.
        self.stop_requested = threading.Event()

        self.step = 0
        self.running = True
        self.done = False
        self.summary: str = ""
        self.thread: threading.Thread | None = None

    def push(self, event_type: str, data: Any) -> None:
        """Enqueue an SSE message; silently drops if the queue is full."""
        payload = json.dumps(
            {"type": event_type, "data": data, "ts": time.time()},
            ensure_ascii=False,
        )
        try:
            self.q.put_nowait(f"data: {payload}\n\n")
        except queue.Full:
            pass  # consumer (browser) too slow — drop rather than block

    def give_feedback(self, text: str) -> None:
        """Inject *text* as the answer to a pending ``ask_user`` call."""
        self.feedback_value = text
        self.feedback_event.set()

    def wait_for_feedback(self, timeout: float = 300.0) -> str:
        """Block until ``give_feedback`` is called or *timeout* expires."""
        self.feedback_event.wait(timeout=timeout)
        self.feedback_event.clear()
        return self.feedback_value


# ── Ada agent wrapper (runs in a background thread) ───────────────────────────

def _run_ada_session(state: SessionState) -> None:
    """Thread target: run Ada and guarantee ``state.done`` is set when finished."""
    try:
        _ada_main(state)
    except Exception as exc:
        state.push("error", str(exc))
    finally:
        state.running = False
        state.done = True
        state.push("done", {"summary": state.summary})


def _ada_main(state: SessionState) -> None:
    """Set up Ada with web-mode hooks, then run the tool loop.

    Key differences from CLI mode
    -----------------------------
    1. Rich console is replaced with ``QueueConsole`` so output goes to SSE.
    2. ``_dispatch`` is monkey-patched to:
       - Check ``stop_requested`` before every tool call.
       - Send ``ask_user`` / ``finish`` events over SSE instead of stdin/stdout.
       - Emit ``tool`` and ``tool_result`` events for the live feed.
    3. The run loop (copied from ``Ada.run``) also checks ``stop_requested``
       at the top of each step so the user can halt mid-loop cleanly.
    """
    from ada.agent import Ada
    from ada.tools import Tools
    from ada.workspace import Workspace
    from ada.llm import LLM, Planner
    from ada.git_ops import Git, GitError

    # --- monkey-patch Console so rich output goes to SSE queue ---
    import io
    from rich.console import Console

    class QueueConsole(Console):
        def print(self, *args, **kwargs):  # type: ignore[override]
            buf = io.StringIO()
            tmp = Console(file=buf, highlight=False, markup=True,
                          width=120, no_color=True)
            tmp.print(*args, **kwargs)
            text = buf.getvalue().strip()
            if text:
                state.push("log", text)
            super().print(*args, **kwargs)

    # Patch the module-level console in agent
    import ada.agent as _agent_mod
    _agent_mod.console = QueueConsole()

    ws = Workspace(state.target_dir)
    llm = LLM(model=state.model or None)
    planner = Planner(model=state.planner_model or None)

    def _planner_fn(prompt: str) -> str:
        return planner.advise(prompt)

    tools = Tools(ws, cmd_timeout=state.cmd_timeout, planner=_planner_fn)

    # Build Ada manually so we can intercept ask_user / finish
    ada = object.__new__(_agent_mod.Ada)
    ada.ws = ws
    ada.tools = tools
    ada.llm = llm
    ada.planner = planner
    ada.max_steps = state.max_steps
    ada.user_goal = state.goal or "(not provided)"
    ada._done_summary = None

    # auto-branch
    git = Git(ws.target_dir)
    if git.is_repo():
        from datetime import datetime
        bname = f"ada/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            git.create_branch(bname)
            state.push("log", f"✓ created working branch: {bname}")
            branch_note = f"working on new branch '{bname}'"
        except GitError as e:
            branch_note = f"on branch {git.current_branch()} (auto-branch failed: {e})"
    else:
        branch_note = "target is NOT a git repo — git_* tools unavailable"

    from ada.agent import _load_prompt
    from ada.tools import TOOL_SCHEMAS
    ada.messages = [
        {"role": "system", "content": _load_prompt()},
        {
            "role": "user",
            "content": (
                f"TARGET_DIR: {ws.target_dir}\n"
                f"USER_GOAL: {ada.user_goal}\n"
                f"WORKER_MODEL: {llm.model}\n"
                f"PLANNER_MODEL: {planner.model}\n"
                f"GIT: {branch_note}\n\n"
                "Begin by executing your Kickoff Protocol (Section 8 of the prompt). "
                "Then enter The Ada Loop. Use the provided tools — do not write code "
                "in plain text answers. After each verified Backlog item, call "
                "`git_commit`. For hard planning calls, use `consult_planner`. "
                "When you need human input, call `ask_user`. "
                "When fully done, call `finish`."
            ),
        },
    ]

    # --- patch _dispatch to intercept ask_user / finish / tool events ---
    original_dispatch = ada._dispatch.__func__  # type: ignore[attr-defined]

    def patched_dispatch(self_inner, name: str, args: dict) -> str:  # type: ignore
        if state.stop_requested.is_set():
            return json.dumps({"error": "stop requested by user"})

        if name == "ask_user":
            question = args.get("question", "")
            state.push("ask_user", question)
            answer = state.wait_for_feedback(timeout=600)
            state.push("user_replied", answer)
            return json.dumps({"user_reply": answer}, ensure_ascii=False)

        if name == "finish":
            summary = args.get("summary", "(no summary)")
            state.summary = summary
            ada._done_summary = summary
            state.push("finish", summary)
            return json.dumps({"acknowledged": True})

        # emit tool event before execution
        state.push("tool", {"name": name, "args": _brief_args(args)})
        result = original_dispatch(ada, name, args)
        try:
            parsed = json.loads(result)
            state.push("tool_result", {"name": name, "result": parsed})
        except Exception:
            pass
        return result

    import types
    ada._dispatch = types.MethodType(patched_dispatch, ada)  # type: ignore

    # --- patched run loop that checks stop_requested ---
    for step in range(1, ada.max_steps + 1):
        if state.stop_requested.is_set():
            state.push("log", "⚠ stop requested — halting Ada loop")
            break

        state.step = step
        state.push("step", step)

        try:
            resp = ada.llm.chat(ada.messages, TOOL_SCHEMAS)
        except Exception as e:
            state.push("error", f"LLM error: {e}")
            break

        msg = resp.choices[0].message
        if msg.content:
            state.push("assistant", msg.content)

        ada.messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        if not msg.tool_calls:
            ada.messages.append({
                "role": "user",
                "content": "Reminder: drive progress via tool calls or call `finish`.",
            })
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = ada._dispatch(name, args)
            ada.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": result,
            })
            if ada._done_summary is not None:
                return

    if not ada._done_summary:
        state.push("log", f"Reached max_steps={ada.max_steps}")


def _brief_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        s = str(v)
        out[k] = s[:120] + "..." if len(s) > 120 else s
    return out


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "web", "index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    body = request.get_json(force=True) or {}
    target_dir = (body.get("target_dir") or "").strip()
    if not target_dir or not Path(target_dir).expanduser().is_dir():
        return jsonify({"error": "target_dir is missing or not a directory"}), 400

    sid = str(uuid.uuid4())
    state = SessionState(
        sid=sid,
        target_dir=str(Path(target_dir).expanduser().resolve()),
        goal=body.get("goal", ""),
        model=body.get("model", os.getenv("ADA_WORKER_MODEL", "")),
        planner_model=body.get("planner_model", os.getenv("ADA_PLANNER_MODEL", "")),
        max_steps=int(body.get("max_steps", os.getenv("ADA_MAX_STEPS", "80"))),
        cmd_timeout=int(body.get("cmd_timeout", os.getenv("ADA_CMD_TIMEOUT", "120"))),
    )
    with _sessions_lock:
        _sessions[sid] = state

    t = threading.Thread(target=_run_ada_session, args=(state,), daemon=True)
    t.name = f"ada-{sid[:8]}"
    state.thread = t
    t.start()

    return jsonify({"session_id": sid})


@app.route("/api/feedback/<sid>", methods=["POST"])
def api_feedback(sid: str):
    with _sessions_lock:
        state = _sessions.get(sid)
    if not state:
        return jsonify({"error": "unknown session"}), 404
    text = (request.get_json(force=True) or {}).get("text", "")
    state.give_feedback(text)
    return jsonify({"ok": True})


@app.route("/api/stream/<sid>")
def api_stream(sid: str):
    with _sessions_lock:
        state = _sessions.get(sid)
    if not state:
        return Response("data: {\"type\":\"error\",\"data\":\"unknown session\"}\n\n",
                        mimetype="text/event-stream")

    def generate():
        while True:
            try:
                chunk = state.q.get(timeout=20)
                yield chunk
            except queue.Empty:
                # heartbeat to keep connection alive
                yield "data: {\"type\":\"ping\"}\n\n"
            if state.done and state.q.empty():
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/stop/<sid>", methods=["POST"])
def api_stop(sid: str):
    with _sessions_lock:
        state = _sessions.get(sid)
    if not state:
        return jsonify({"error": "unknown session"}), 404
    state.stop_requested.set()
    # unblock any waiting ask_user
    state.give_feedback("[user stopped]")
    return jsonify({"ok": True})


@app.route("/api/status/<sid>")
def api_status(sid: str):
    with _sessions_lock:
        state = _sessions.get(sid)
    if not state:
        return jsonify({"error": "unknown session"}), 404
    return jsonify({
        "running": state.running,
        "done": state.done,
        "step": state.step,
        "summary": state.summary,
    })


@app.route("/api/models")
def api_models():
    """Return available model names by querying the Ollama /v1/models endpoint.

    Falls back to an empty list with an error message if Ollama is unreachable
    (e.g. ``ollama serve`` is not running).  The frontend shows text inputs
    instead of dropdowns in that case.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    try:
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = sorted(m["id"] for m in data.get("data", []))
        return jsonify({"models": models})
    except Exception as exc:
        return jsonify({"models": [], "error": str(exc)})


@app.route("/api/browse")
def api_browse():
    """Return subdirectories of *path* for the directory-picker modal.

    Query parameters
    ----------------
    path : str
        Absolute path to browse.  Defaults to the user's home directory.
    up : "1" | "0"
        If "1", navigate to the parent of *path* first.

    Response JSON
    -------------
    {
      "current": "/abs/path",
      "parent":  "/abs/parent" | null,   # null at filesystem root
      "dirs":    [{"name": "foo", "path": "/abs/path/foo", "has_sub": true}, ...],
      "hidden":  [...],                  # dot-dirs, same structure
    }
    """
    raw = request.args.get("path", "").strip() or str(Path.home())
    go_up = request.args.get("up", "0") == "1"
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        p = p.parent if p.parent.is_dir() else Path.home()
    if go_up and p.parent != p:
        p = p.parent

    try:
        dirs = sorted(
            [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")],
            key=lambda x: x.name.lower(),
        )
        hidden = sorted(
            [c for c in p.iterdir() if c.is_dir() and c.name.startswith(".")],
            key=lambda x: x.name.lower(),
        )
    except PermissionError:
        dirs, hidden = [], []

    def _entry(c: Path) -> dict:
        try:
            has_sub = any(True for x in c.iterdir() if x.is_dir())
        except PermissionError:
            has_sub = False
        return {"name": c.name, "path": str(c), "has_sub": has_sub}

    parent = str(p.parent) if p != p.parent else None
    return jsonify({
        "current": str(p),
        "parent": parent,
        "dirs": [_entry(c) for c in dirs],
        "hidden": [_entry(c) for c in hidden],
    })


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Gracefully shut down the server."""
    def _kill():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"ok": True, "message": "Server shutting down…"})


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ada Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  Ada Web UI  →  http://{args.host}:{args.port}\n")
    # threaded=True required for SSE + concurrent requests
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
