# Ada — Architecture

This document describes the internal design of the Ada runtime.
For user-facing documentation see [README.md](README.md).

---

## Package overview

```
ada/
├── __init__.py    public API (re-exports Ada class)
├── agent.py       tool-calling loop, Ada class, context compaction
├── llm.py         LLM client (Worker + Planner) + provider profiles
├── tokens.py      tokenizer wrapper (tiktoken with safe fallback)
├── tools.py       62 tool implementations + OpenAI function schemas
├── semantic.py    tree-sitter wrapper: multi-language nav + structural edit
├── embed.py       embedding-based vector search (sqlite + OpenAI/-compat API)
├── mcp.py         MCP (Model Context Protocol) stdio client
├── verify.py      auto-detect & run pytest/ruff/mypy/eslint/tsc/cargo/...
├── safety.py      dangerous-command guard + secrets scan
├── conventions.py loads AGENTS.md / CLAUDE.md into kickoff prompt
├── pricing.py     per-model USD cost estimator
├── audit.py       JSONL audit log of every tool call
├── replay.py      pretty-print a saved audit log
├── budget.py      hard caps on cost/steps/tokens/wall-clock
├── checkpoint.py  filesystem snapshot/rollback
├── recall.py      cross-session memory recall (journal/backlog/audit)
├── traceback_parser.py  pytest/unittest traceback → frames
├── refactor.py    word-boundary bulk rename across whitelisted file types
├── web.py         allowlisted HTTPS GET + HTML→text reduction
├── impact.py      changed-files → affected tests + diff stats/risk verdict
├── tasks.py       JSON-backed multi-goal task queue
├── notebook.py    Jupyter (.ipynb) read + per-cell edit
├── memstore.py    SQLite-backed cross-run K/V memory
├── profile.py     cProfile snippet runner with hot-funcs report
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

### Context compaction

Before each LLM call, `Ada._maybe_compact()` checks the live prompt size via
`ada/tokens.py` (which uses **tiktoken** when available and a `len/3` fallback
otherwise). When token usage exceeds `ADA_COMPACT_TOKENS` (default 24000) —
or the safety-net message count `ADA_COMPACT_AFTER` (default 200) — the middle
slice (everything between the kickoff turn and the last `ADA_KEEP_RECENT`
messages, default 20) is fed to the **planner** model along with any prior
summary. The transcript itself is hard-capped at half the token budget so the
planner's own prompt never blows up; the oldest entries are elided first.
The planner returns a structured markdown digest (project understanding,
backlog state, recent commits, verification results, open questions, important
file excerpts) which replaces the slice as a single `user` message tagged
`[CONTEXT SUMMARY OF EARLIER STEPS]`. The boundary is walked forward past any
orphan `tool` messages so an assistant→tool pair is never split.

This keeps the live prompt bounded regardless of session length, and the
summary is regenerated (folding the previous one in) whenever the threshold
is hit again.

### Provider profiles & token accounting

`ada/llm.py` exposes a `PROFILES` dict keyed by provider name
(`ollama`, `openai`, `deepseek`, `moonshot`, `anthropic`). When `ADA_PROFILE`
is set (or `--profile` is passed on the CLI) `_apply_profile()` populates
env defaults *without* overriding existing values, so explicit env vars and
flags always win. Each profile names its own `*_API_KEY` env var so multiple
keys can coexist in one `.env`.

Both `LLM` and `Planner` accumulate `usage` counters (prompt / completion /
total tokens, request count) on every successful call. `Ada._print_usage()`
emits a one-line summary at the end of every run, supporting head-to-head
model comparison.

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

## Semantic engine (ada/semantic.py)

A thin layer over `tree-sitter-language-pack` that powers four agent tools
(`list_symbols`, `find_definition`, `find_references`, `ts_edit`) for every
language Ada supports beyond Python.

### Language registry

```python
@dataclass(frozen=True)
class _LangSpec:
    name: str                       # "python", "javascript", ...
    def_types: tuple[str, ...]      # nodes that count as a "definition"
    name_fields: tuple[str, ...]    # how to extract the symbol name
    container_types: tuple[str, ...] # nodes whose body holds nested defs

_LANGS: dict[str, _LangSpec] = {".py": ..., ".js": ..., ...}
```

Adding a new language is a one-line entry in `_LANGS` once you know the
relevant tree-sitter node types.

### Edit validation

`Semantic.edit_symbol()` always:

1. Re-parses the file *before* the edit and counts `ERROR`/`MISSING` nodes
   (`baseline`).
2. Splices in the new code (UTF-8 byte aware via `start_byte`/`end_byte`).
3. Re-parses the result. If new error count > baseline → **rollback**, raise
   `ValueError("edit would introduce N new parse error(s); aborted")`.

This guarantees a successful `ts_edit` never leaves the file in a worse
syntactic state than it was found.

### Indentation

`replace` and `insert_before` splice at the symbol's exact column, so only
lines 2..N are reindented (`_reindent_tail`). `insert_after` splices at
end_byte (column 0 after a newline), so all lines get the indent prefix
(`_reindent`). This avoids the classic "double-indent on first line" bug.

### Limitations

* **Not scope-aware** — `find_references` reports every identifier-position
  occurrence of a name; local-variable shadows still appear. Closing this
  gap requires a real LSP client (out of scope).
* **Optional dependency** — if the parser pack isn't installed, every method
  on `Semantic` returns `False` from `available()`; `Tools` short-circuits
  with a structured error.

---

## Verification engine (ada/verify.py)

A small marker-file based detector for the standard
**test / lint / format** toolchains plus a runner that captures their
output for the LLM:

```python
Verifier(root).detect()       # → {"tests": [...], "lint": [...], "format": [...]}
Verifier(root).run_tests()    # runs every detected test toolchain in series
Verifier(root).run_lint()     # likewise for lint
```

Detection rules:

| Tool | Marker(s) |
|------|-----------|
| `pytest`   | `pytest.ini`, `tests/`, `[tool.pytest.ini_options]` |
| `ruff`     | `ruff.toml`, `[tool.ruff]` in `pyproject.toml` |
| `mypy`     | `mypy.ini`, `[tool.mypy]` |
| `black`    | `[tool.black]` |
| `eslint`   | `.eslintrc*`, `eslint.config.*` |
| `tsc`      | `tsconfig.json` |
| `go test`/`vet` | `go.mod` |
| `cargo test`/`clippy` | `Cargo.toml` |
| `npm test` | `package.json` |

Per-tool `subprocess.run` with timeout + tail-truncated output. Polyglot
repos (Python + Go + JS in one tree) get all matching tools run.

## Safety guards (ada/safety.py)

Two narrow filters chosen for **high precision over high recall**:

* `assess_command(cmd)` — small high-signal blacklist of regexes
  (`rm -rf /|.|~|*`, `git push --force`, `curl … | sh`, `DROP TABLE`,
  `dd of=/dev/sd*`, fork bombs, ...). Returns `CommandRisk(is_dangerous,
  reason)`. `Tools.run_command` consults it before every shell call;
  blocked commands return a structured "blocked" dict instead of throwing
  so the LLM can see the rule that fired and ask the user.
* `scan_secrets(text)` — pattern-based detector for AWS/GitHub/OpenAI/
  Anthropic/Google/Slack/Stripe keys, JWTs, and PEM-encoded private keys.
  Generic high-entropy detection is intentionally **not** included
  (commit hashes, base64 fixtures cause too many false positives).

Both honor `ADA_SAFETY=0` for opt-out.

## Conventions loader (ada/conventions.py)

Mirrors Claude Code / Aider behavior: at startup the agent scans the
project root for `AGENTS.md`, `CLAUDE.md`,
`.ada/CONVENTIONS.md`, `.ada/conventions.md`,
`.github/copilot-instructions.md`. Anything found is concatenated (8 KB
per-file cap, 16 KB total cap) and inserted into the kickoff user message
so the model sees project rules on every request without spending tools
on them.

## Repo map (Semantic.repo_map)

After bootstrapping, Ada runs `Semantic.repo_map(root)` and renders it
into the same kickoff message. Files are scored by depth (shallower = more
important) minus a test-folder penalty, then capped at
`ADA_REPO_MAP_FILES`. The rendered text is hard-capped at
`ADA_REPO_MAP_CHARS` to bound prompt growth.

## Pricing (ada/pricing.py)

A small static USD/M-token table for the major providers (OpenAI,
Anthropic, DeepSeek, Moonshot, Qwen). Local backends (`ollama:`,
`qwen3.5:*`, `llama:*`, etc.) price at $0. Prefix matching means
`gpt-4o-2024-11-20` resolves to the `gpt-4o` row without a per-version
entry. `pricing.set_price()` lets users override at runtime.

`Ada._print_usage()` calls `estimate_cost()` for both the worker and the
planner and prints both alongside token counts and per-step timings.

## Embedding search (ada/embed.py)

Persistent vector index for the project, used to answer "*where is the
code that does X?*" when literal grep would miss it.

* **Storage** — SQLite at `.ada/embeddings.db`. Schema:
  `chunks(id, path, start_line, end_line, text, vector BLOB)` plus a
  `files(path, mtime, size, model)` table for incremental skips.
  Vectors are stored as raw little-endian float32 bytes.
* **Embedder backends** — `_OpenAIEmbedder` (any
  OpenAI-compatible `/v1/embeddings` endpoint, default
  `text-embedding-3-small`) or `_FakeEmbedder` (deterministic SHA-256
  hash, used by tests when `ADA_EMBED_FAKE=1`).
* **Chunking** — symbol-aware via tree-sitter when available
  (one chunk per function/class), otherwise overlapping line windows
  (40 lines / 10-line overlap, ≤ 4 KB per chunk).
* **Search** — pure-Python cosine over normalised vectors. Adequate up
  to ~50 k chunks; larger repos should swap in `sqlite-vec` or FAISS.
* **Bootstrap** — `Tools.semantic_search` auto-builds the index on
  first call so the model doesn't need a separate "index now" step.

When the embedder backend is unavailable (no API key + no fake flag),
`available()` returns `False` and the tool returns a structured
"embeddings not configured" error instead of throwing.

## MCP client (ada/mcp.py)

Tiny, stdlib-only stdio client for the
[Model Context Protocol](https://modelcontextprotocol.io). Reads
`.ada/mcp.json`, spawns each declared server as a subprocess, runs the
JSON-RPC 2.0 handshake (`initialize` → `notifications/initialized` →
`tools/list`), and exposes the discovered tools to Ada.

* **Tool namespacing** — each server tool is registered as
  `mcp_<server>_<tool>` (non-alnum chars replaced with `_`,
  capped at 64 chars to satisfy the OpenAI function-name regex).
* **Threading** — one background reader thread per server demuxes
  responses by JSON-RPC `id` into `threading.Event`s.
* **Failure isolation** — a server that fails to spawn or initialise is
  recorded in `MCPClient.errors` but never blocks the others.
* **Schema injection** — `Ada.run()` rebuilds the merged schema list
  (`TOOL_SCHEMAS + mcp.tool_schemas()`) every step, so live tool
  catalogues stay in sync.
* **Dispatch routing** — `Ada._dispatch` routes any name beginning
  with `mcp_` and present in `MCPClient.tool_map` to the server;
  everything else falls through to the static `Tools` surface.
* **Cleanup** — `_shutdown()` terminates all subprocesses on either
  natural finish or step-cap exit.

---

## End-to-end evaluation (evals/)

Reproducible harness that pits Ada against fixture tasks and reports
pass/fail. Lives outside the `ada/` package so it's never imported at
runtime.

```
evals/
├── run_evals.py         CLI: discover, prepare, run, verify, report
└── tasks/
    ├── fix_bug_stats/   one-line bug fix
    ├── add_uppercase/   implement a missing function
    ├── rename_class/    cross-file rename
    ├── find_retry/      semantic_search recall test
    └── verify_then_fix/ run-tests-first behaviour
```

Each task directory has `task.json` (goal + verify cmd + budget) and
`verify.sh` (returns 0 = pass). `run_evals.py` copies sources into a
fresh tempdir, `git init`s a baseline, runs `main.py` non-interactively
(stdin closed so `ask_user` auto-returns ""), then runs `verify.sh`.

`tests/test_evals_harness.py` exercises discovery, sandbox prep, and
the verify-only path without invoking the LLM, so the harness itself is
CI-safe.

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
