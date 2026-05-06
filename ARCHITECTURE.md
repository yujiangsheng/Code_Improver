# Ada — Architecture

This document describes the internal design of the Ada runtime.
For user-facing documentation see [README.md](README.md).

---

## Package overview

```
ada/
├── __init__.py    public API (re-exports Ada class)
├── agent.py       tool-calling loop, Ada class
├── llm.py         LLM client (Worker + Planner two-tier routing)
├── tools.py       18 tool implementations + OpenAI function schemas
├── workspace.py   .ada/ workspace manager + path sandbox
└── git_ops.py     Git subprocess wrapper
```

Entry points:

| Entry point | Audience |
|-------------|----------|
| `main.py` | CLI users |
| `server.py` | Web UI (Flask + SSE) |
| `from ada import Ada` | Library consumers |

---

## LLM two-tier routing

```
Worker model  (every step)
  • qwen3.5:9b  (default)
  • tool-calling enabled
  • drives the loop
       │
       │ calls consult_planner tool
       ▼
Planner model  (on demand)
  • qwen3.5:27b  (default)
  • no tool calls — text-only advice
  • used for backlog ranking, design decisions
```

Both models share the same `OPENAI_BASE_URL` and communicate via the
`openai.OpenAI` client with streaming completion (deltas merged before
passing to the tool loop).

Configuration priority (highest first):

```
CLI flag  →  environment variable  →  hardcoded default
```

See `ada/llm.py` for `DEFAULT_WORKER_MODEL` and `DEFAULT_PLANNER_MODEL`.

---

## Agent loop (ada/agent.py)

```
Ada.run()
│
├─ build system prompt (prompt.md + workspace artifacts)
│
└─ loop until finish / stop / max_steps:
    │
    ├─ [1] call Worker LLM  (stream=True, tools=TOOL_SCHEMAS)
    │        collect assistant message + tool_calls
    │
    ├─ [2] if no tool_calls:
    │        nudge LLM with "you must call a tool", continue
    │
    └─ [3] dispatch each tool_call:
             │
             ├─ ask_user  → prompt human (stdin in CLI, SSE event in web)
             ├─ finish    → store summary, stop loop
             └─ others    → Tools._dispatch(name, args) → result string
```

### Special tool handling

`ask_user` and `finish` are intercepted before reaching `Tools._dispatch`:
- In CLI mode: interact with stdin/stdout directly.
- In web mode: `_ada_main()` monkey-patches `_dispatch` to emit SSE events
  and block on `threading.Event` instead of stdin.

### Nudge on empty tool calls

If the LLM returns a content-only message without tool calls, Ada appends a
nudge message to the history and retries immediately without incrementing the
step counter. This prevents the loop from stalling when the model "thinks out
loud" instead of acting.

---

## Workspace (ada/workspace.py)

```
<target_dir>/
└── .ada/
    ├── profile.md        Phase 0: project overview
    ├── mental_model.md   Phase 1: module map, data flow
    ├── baseline.json     Phase 2: numeric baseline metrics
    ├── backlog.md        Phase 3: prioritised improvement table
    ├── journal.md        per-iteration entries (append-only)
    ├── metrics.csv       time-series CSV (timestamp, phase, metric, value)
    └── questions.md      open questions accumulated during the run
```

`Workspace.resolve(path)` is the single sandbox enforcement point: it calls
`Path.resolve()` and verifies the result sits inside `target_dir`. Any
attempt to escape raises `PermissionError`.

---

## Tools (ada/tools.py)

### Design principles

1. **Sandboxed** — every file/shell tool calls `workspace.resolve()` first.
2. **Return dicts** — tools return `dict[str, str]` so the LLM gets JSON.
3. **Raise on hard errors** — callers format exceptions as `{"error": "..."}`.
4. **Truncate large output** — shell stdout > 8 KB is tail-truncated;
   `read_file` reads ≤ 800 lines; `git_diff` caps at 12 KB.
5. **Skip binary files** — `grep` uses a `\x00`-in-first-1KB heuristic.

### Tool dispatch

```python
Tools._dispatch(name: str, args: dict) -> str
```

Looks up `name` in a `dict[str, Callable]` built once at construction time.
Returns a JSON-formatted string on success, `{"error": "..."}` on failure.

---

## Git operations (ada/git_ops.py)

All Git interactions use `subprocess.run(["git", ...])`. No gitpython dependency.

### Branch strategy

```
main  ─●───────────────────────────── (untouched)
        \
         ● ada/20250518-143022
            auto-created on launch
            one commit per verified Backlog item
```

`GitOps.auto_branch()` creates `ada/<YYYYMMDD-HHmmSS>` and checks it out.
Disabled with `--no-auto-branch`.

### Diff cap

`git_diff` hard-truncates at `_DIFF_CAP = 12_000` chars to protect LLM context.

---

## Web server (server.py)

### Session lifecycle

```
Browser                  Flask                   Ada thread
  │                        │                         │
  │  POST /api/start       │  create SessionState     │
  ├───────────────────────►│  spawn daemon thread ───►│
  │  {session_id}          │                         │ running
  │◄───────────────────────┤                         │
  │                        │                         │
  │  GET /api/stream/<sid> │                         │
  ├───────────────────────►│                         │
  │  text/event-stream     │   state.q (queue)       │
  │  data:{type:step}      │◄────────────────────────┤
  │  data:{type:tool}      │◄────────────────────────┤
  │  data:{type:ask_user}  │◄────────────────────────┤ blocked
  │◄───────────────────────┤                         │
  │  POST /api/feedback    │  give_feedback(text)    │
  ├───────────────────────►├────────────────────────►│ unblocked
  │  data:{type:finish}    │◄────────────────────────┤
  │  data:{type:done}      │◄────────────────────────┤
  │◄───────────────────────┤                         │
```

### SSE heartbeat

The SSE generator sends a `ping` event every 20 seconds when the queue is
empty, preventing proxies from closing idle connections.

### Console monkey-patching

`_ada_main()` replaces `ada.agent.console` with a `QueueConsole` whose
`print()` serialises Rich renderables to plain text and pushes them into
`state.q` as `log` events — the browser gets the same output the CLI prints.

### Model picker endpoint

`GET /api/models` queries Ollama's `/v1/models` and returns a sorted list of
model IDs. The frontend populates worker/planner `<select>` dropdowns from
this list. A "自定义模型…" sentinel switches to a free-text `<input>`.

---

## Data flow summary

```
User goal
    │
    ▼
Ada.run()
    │  reads prompt.md + .ada/ artifacts → system prompt
    │
    ├─► Worker LLM: decide next action
    │       │
    │       └─► tool_calls[]
    │               │
    │               ├─ read_file / grep / list_dir   → content string
    │               ├─ write_file / edit_file         → diff summary
    │               ├─ run_command                    → stdout/stderr
    │               ├─ git_*                          → git output
    │               ├─ update_artifact / append_*     → workspace writes
    │               ├─ consult_planner                → Planner LLM text
    │               ├─ ask_user                       → human reply (blocks)
    │               └─ finish                         → summary (exit)
    │
    └─► repeat until finish / max_steps / stop_requested
```
