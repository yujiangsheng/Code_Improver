# Ada — Autonomous Code-Improvement Agent

Ada reads a directory of code, understands its purpose, and iteratively
improves it through a `Comprehend → Baseline → Plan → Improve → Verify →
Reflect → Report` loop until the user is satisfied.

The agent's behaviour spec lives in [prompt.md](prompt.md). This package is the
runtime that executes that spec via an LLM + tool-calling loop.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # defaults already point at local Ollama
```

**Default backend: local [Ollama](https://ollama.com).** No API key required.
Make sure the daemon is up and the models are pulled:

```bash
ollama serve &                 # if not already running
ollama pull qwen3.5:9b         # default worker model  (fast, code-capable)
ollama pull qwen3.5:27b        # default planner model (stronger reasoning)
```

To use a hosted provider instead, edit `.env`:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxx
ADA_WORKER_MODEL=gpt-4o-mini
ADA_PLANNER_MODEL=gpt-4o
```

Any **OpenAI-compatible** Chat Completions endpoint with function calling works
(OpenAI, DeepSeek, Moonshot, vLLM, Ollama, ...).

## Run

```bash
python main.py /path/to/your/project --goal "让所有测试通过并把 lint 警告清零"
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--goal TEXT` | _(ask Ada)_ | High-level improvement objective |
| `--model MODEL` | `$ADA_WORKER_MODEL` | Worker LLM name |
| `--planner MODEL` | `$ADA_PLANNER_MODEL` | Planner LLM name |
| `--steps N` | `80` | Max tool-loop iterations |
| `--cmd-timeout SEC` | `120` | Per-shell-command timeout |
| `--no-auto-branch` | — | Stay on current branch |

## What Ada writes

A persistent workspace is created at `<target>/.ada/`:

```
.ada/
├── profile.md          project profile (Phase 0)
├── mental_model.md     module map / data flow (Phase 1)
├── baseline.json       baseline metrics
├── backlog.md          prioritised improvement list
├── journal.md          per-iteration log
├── metrics.csv         metric time series
└── questions.md        open questions for the human
```

Artifacts survive across sessions — Ada re-reads them on the next launch.

---

## Web UI

Ada ships with a single-page application for a richer experience.

```bash
python server.py --port 7878
# then open http://127.0.0.1:7878
```

**Features:** live SSE feed, worker/planner model dropdowns (auto-populated from
Ollama), directory picker modal, `ask_user` feedback panel, Stop button.

---

## Tool catalogue

All file and shell tools are sandboxed inside the target directory.

| Tool | Category | Description |
|------|----------|-------------|
| `list_dir` | Read | List directory entries; dirs suffixed with `/` |
| `read_file` | Read | Read file with line numbers (<=800 lines/call) |
| `grep` | Read | Regex search across all text files |
| `write_file` | Write | Create or overwrite a file |
| `edit_file` | Write | Replace exactly one occurrence of a string |
| `ast_edit` | Write | Structural Python edit by symbol name (replace / delete / insert / append / show) — preserves formatting and validates syntax |
| `patch_apply` | Write | Apply a unified-diff (`git diff`) patch atomically across one or more files; ±200 lines of drift tolerated, supports create/delete via `/dev/null`, dry-run via `check_only` |
| `list_symbols` | Semantic | List every named def in a source file (py / js / ts / tsx / go / rs / java / c / cpp …) via tree-sitter |
| `find_definition` | Semantic | Project-wide jump-to-definition for `name` (no false positives from strings/comments) |
| `find_references` | Semantic | Project-wide identifier-position references; ignores strings/comments (NOT scope-aware) |
| `ts_edit` | Semantic | Cross-language counterpart to `ast_edit` (replace / delete / insert_before / insert_after / show); validates parse and rolls back on syntax error |
| `read_symbol` | Semantic | Return the source of one function/class/method (~10× cheaper than `read_file`) |
| `repo_map` | Semantic | Compact outline of project's key symbols (auto-injected at startup; explicit call refreshes) |
| `run_tests` | Verify | Auto-detect and run pytest / unittest / go test / cargo test / npm test |
| `run_lint` | Verify | Auto-detect and run ruff / mypy / eslint / tsc / go vet / clippy |
| `detect_toolchain` | Verify | List which test/lint/format tools are detected |
| `scan_secrets` | Safety | Scan a file or whole repo for AWS / GitHub / OpenAI / private-key leaks |
| `semantic_search` | Embedding | Find code by *meaning* — surfaces "retry/backoff" even when the word "retry" doesn't appear |
| `reindex_embeddings` | Embedding | Refresh the vector index after large refactors |
| `embed_stats` | Embedding | Index size + backend status |
| `mcp_status` | MCP | List configured MCP servers and the `mcp_<server>_<tool>` calls they expose |
| `run_command` | Shell | Execute a shell command (output tail-truncated; dangerous patterns blocked unless `force=True`) |
| `git_status` | Git | Show branch + short status |
| `git_diff` | Git | Working-tree or staged diff (capped at 12 KB) |
| `git_create_branch` | Git | Create and check out a branch |
| `git_commit` | Git | Stage all changes and commit |
| `git_revert` | Git | Hard-reset to a ref (**destructive**) |
| `update_artifact` | Workspace | Overwrite a `.ada/` artifact file |
| `append_journal` | Workspace | Append a timestamped entry to `journal.md` |
| `append_metric` | Workspace | Append a row to `metrics.csv` |
| `read_artifact` | Workspace | Read a `.ada/` artifact |
| `consult_planner` | Multi-model | Ask the stronger planner model for advice |
| `ask_user` | Control | Pause and ask the human for input |
| `finish` | Control | Declare task complete with a final summary |

## Multi-model routing

Ada uses a two-tier architecture: a fast **Worker** drives the tool loop every
step; a stronger **Planner** is called on demand via `consult_planner`.

```bash
# Local Ollama
export ADA_WORKER_MODEL=qwen3.5:9b
export ADA_PLANNER_MODEL=qwen3.5:27b

# Or per-run via CLI
python main.py . --model qwen3.5:9b --planner qwen3-coder:30b
```

Recommended combinations:

| Worker | Planner | Speed | Quality |
|--------|---------|-------|---------|
| `qwen3.5:9b` | `qwen3.5:27b` | Fast | Good |
| `qwen3.5:9b` | `qwen3-coder:30b` | Medium | Better |
| `qwen3-coder:30b` | `qwen3-coder:30b` | Slow | Best |

## Git workflow

On launch, Ada auto-creates a working branch `ada/<YYYYMMDD-HHMMSS>` so your
main branch stays clean. After every verified Backlog item, Ada calls
`git_commit`. To inspect or revert:

```bash
git log --oneline ada/<ts>
git diff main..ada/<ts>
git checkout main           # discard everything Ada did
```

Disable auto-branch with `--no-auto-branch` (Ada will still commit on the
current branch).

## Semantic tooling (tree-sitter)

`grep` finds substrings; `ast_edit` only handles Python. To navigate and
mutate large multi-language repos precisely, Ada ships four **tree-sitter**
backed tools:

| Tool | What it returns |
|------|-----------------|
| `list_symbols(path)` | Every named def in one file with dotted `qualified_name` (e.g. `Calc.Inner.deep`) |
| `find_definition(name, path=".")` | Project-wide list of definitions whose leaf name == `name` |
| `find_references(name, path=".")` | Identifier-position occurrences only — strings/comments excluded |
| `ts_edit(path, operation, symbol, code)` | Replace / delete / insert_before / insert_after / show a symbol; reparses and rolls back on new syntax errors |

Supported extensions: `.py .js .mjs .cjs .jsx .ts .tsx .go .rs .java .c .h
.cpp .cc .cxx .hpp .hh`. Other files raise an explicit "unsupported
language" error.

The two parser packages (`tree-sitter`, `tree-sitter-language-pack`) are
**optional** — if missing, all four tools return a structured
`{"error": "tree-sitter not available …"}` and the rest of Ada keeps working.

```bash
pip install tree-sitter tree-sitter-language-pack   # one-shot install
```

**Limitation**: `find_references` is *not* scope-aware. A local variable
that shadows the queried name will still appear in the result. This is the
gap an LSP client would close; for now, prefer `find_references` over `grep`
for navigation, but verify each hit before bulk-renaming.

## Verification loop (tests + lint)

Ada auto-detects the project's toolchain at startup. Each tool runs only if
its marker file is present; nothing detected = silent no-op.

| Category | Tools detected |
|----------|----------------|
| tests    | `pytest`, `unittest`, `go test`, `cargo test`, `npm test` |
| lint     | `ruff`, `mypy`, `eslint`, `tsc --noEmit`, `go vet`, `clippy` |
| format   | `black --check`, `prettier --check`, `gofmt -l` |

```python
tools.run_tests()                       # run every detected suite
tools.run_lint(only=["ruff"])           # subset
tools.detect_toolchain()                # list what would run
```

The agent prompt encourages the model to call `run_tests` and `run_lint`
between every meaningful edit and `git_commit`, closing the
**edit → verify → commit** loop without the user having to nag.

## Repo conventions auto-load

Every Ada run scans the project root for any of the following and
concatenates them into the kickoff prompt — the model sees them on every
request without you restating the rules:

- `AGENTS.md`
- `CLAUDE.md`
- `.ada/CONVENTIONS.md` / `.ada/conventions.md`
- `.github/copilot-instructions.md`

Per-file cap: 8 KB. Total cap: 16 KB. Drop a single `AGENTS.md` for
project-wide rules.

## Repo map

When tree-sitter is installed, Ada generates a compact outline (file →
top-level symbols + line ranges, ranked by depth and density) at startup
and injects it into the kickoff message. Tune via env vars:

```bash
export ADA_REPO_MAP_FILES=60      # max files in outline
export ADA_REPO_MAP_SYMS=10       # max symbols per file
export ADA_REPO_MAP_CHARS=4000    # hard char cap on rendered text
```

Call `repo_map` mid-run to refresh after large refactors.

## Safety guards

`run_command` screens every command against a small high-signal blacklist
(`rm -rf /`, `sudo`, `git push --force`, `curl … | sh`, `DROP TABLE`,
filesystem `dd`, fork bombs, …) and returns
`{"blocked": True, "reason": ..., "hint": ...}` instead of executing.

* **Override per call**: pass `force=True` after explicit human
  confirmation via `ask_user`.
* **Disable globally**: `export ADA_SAFETY=0`.

`scan_secrets` detects AWS keys, GitHub PATs, OpenAI / Anthropic / Google
API keys, JWTs, and private-key blocks. Run before every commit.

## Observability

After every run Ada prints:

```
tokens — worker[gpt-4o-mini]: 12,341 total (10,221 in / 2,120 out, 18 req, cost $0.0028);
        planner[claude-3-5-sonnet]: 4,500 total (3 req, cost $0.0420)
time   — 87.3s total, 18 steps, avg 4.85s/step
slowest tools — run_tests(2x, 14.1s), read_file(8x, 3.2s), grep(5x, 2.1s)
```

Cost prices are in `ada/pricing.py`; override at runtime with
`pricing.set_price(prefix, in_per_M, out_per_M)`. Local models (Ollama,
qwen3.5, llama, etc.) price at $0.

## Self-critique gate

Set `ADA_SELF_CRITIQUE=1` to make Ada perform one mandatory self-review
before `finish` succeeds: did it actually verify the change, are obvious
edge cases handled, does the diff match the goal? The first `finish`
attempt is deflected with a critique prompt; the second succeeds.

## Embedding-based semantic search

`grep` only finds literal text; `semantic_search` finds **meaning**.

```python
tools.semantic_search("where is the retry / backoff logic")
# → hits in network.py even if the function is called sleep_and_redo
```

* **Backend**: any OpenAI-compatible `/v1/embeddings` endpoint.
  Defaults to `text-embedding-3-small`. Override with
  `ADA_EMBED_MODEL`, `ADA_EMBED_BASE_URL`, `ADA_EMBED_API_KEY`.
* **Storage**: `.ada/embeddings.db` (SQLite, raw float32 vectors).
* **Incremental**: re-indexing skips files whose `(mtime, size)` matches.
* **Symbol-aware**: chunks follow function/class boundaries when
  tree-sitter is available; falls back to overlapping line windows.
* **Offline tests**: `ADA_EMBED_FAKE=1` uses a deterministic hash-based
  embedder (no API calls) — used by `tests/test_embed.py`.

If neither `OPENAI_API_KEY` nor `ADA_EMBED_FAKE` is set, the tool
returns a clear "embeddings not configured" error.

## MCP (Model Context Protocol)

Drop a `.ada/mcp.json` into your project to plug in any MCP server —
filesystem, GitHub, Linear, Postgres, etc. Tools are auto-discovered
and merged into Ada's tool surface as `mcp_<server>_<tool>`.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/repo"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
    }
  }
}
```

Use `mcp_status` to see what was loaded; failed servers are recorded in
the `errors` array but never break startup.

## End-to-end evaluation

The `evals/` package runs Ada non-interactively against a set of
fixture tasks and reports pass/fail. Use it after every prompt or model
change to catch regressions.

```bash
python -m evals.run_evals --list           # show 5 shipped tasks
python -m evals.run_evals --task fix_bug_stats
python -m evals.run_evals --report out.json
```

Each task lives in `evals/tasks/<name>/` with:

* `task.json` — `{"goal", "verify_cmd", "max_steps", "timeout_sec"}`
* `verify.sh` — shell script returning 0 = pass
* source files for the project under test

Currently shipped tasks: `fix_bug_stats`, `add_uppercase`,
`rename_class`, `find_retry`, `verify_then_fix`. Add new ones by
dropping in another directory.

## Audit log & trajectory replay

Every tool call is appended as one JSON line to `.ada/audit.jsonl`
(step, tool, args/result preview, ok flag, duration). Replay any past
run with:

```bash
python -m ada.replay .ada/audit.jsonl              # timeline + summary
python -m ada.replay .ada/audit.jsonl --summary    # aggregate only
python -m ada.replay .ada/audit.jsonl --tool run_tests
```

Disable with `ADA_AUDIT=0`.

## Hard budget caps

Stop runaway runs before they burn money or time. Any subset of these
env vars trips a graceful shutdown (mid-step is never interrupted):

| Variable | Meaning |
| --- | --- |
| `ADA_MAX_COST` | total USD across worker + planner |
| `ADA_MAX_STEPS_HARD` | secondary step cap |
| `ADA_MAX_TOKENS` | combined prompt + completion tokens |
| `ADA_MAX_SECONDS` | wall-clock since run start |

Unset = unlimited on that axis. Trigger emits a red `BUDGET EXCEEDED`
panel and the usage summary.

## Read-file cache

Repeat reads of the same `(path, mtime, size, start_line, end_line)`
return the cached body plus `{"cached": True, "hint": "..."}`, nudging
the model to vary its window or move on. The cache invalidates
automatically when the file changes (mtime/size moves) and is per-run.

## Failure locator & batch edits

* `locate_failures` parses a pytest/unittest traceback and returns each
  in-workspace frame as `{file, line, symbol, preview}` (±5 lines), so
  the model can fix bugs without an extra `read_file` round-trip.
* `batch_edit` applies many `edit_file` operations transactionally — any
  failure rolls back every prior edit in the batch. Use it for
  cross-file renames or coordinated changes.

## Write-time secret guard

`write_file` / `edit_file` / `batch_edit` scan the *new* content for
credential patterns (AWS keys, GitHub PATs, OpenAI/Anthropic keys, JWTs,
private-key blocks). If anything matches, the write is refused with a
structured error listing the hits.

* Override per-call: pass `allow_secrets=true` (only do this for fixtures).
* Disable globally: `ADA_GUARD_SECRETS=0`.

## Checkpoints (snapshot / rollback)

`create_checkpoint`, `restore_checkpoint`, `list_checkpoints` give Ada a
filesystem-level save point that works even when the project isn't a git
repo. `safe_run` wraps a single tool call in an auto-checkpoint and
rolls back on error — useful for speculative refactors.

## Cross-session recall

At startup Ada appends a "prior-session recall" block to the system
prompt summarising the recent journal, current backlog, and the last
few audit entries from the previous run. First runs are unaffected
(empty when no `.ada/` state exists).

## Plan mode

`make_plan` asks the planner LLM for a structured plan
(Goal/Assumptions/Steps/Risks/DoD) and persists it to `.ada/plan.md`.
Useful before tackling complex multi-step tasks.

## Self-introspection

`ada_config` reports tool count, env caps, embedding/MCP availability,
and read-cache stats so the agent can audit its own configuration.

## Refactor, focused tests, web fetch

* `rename_symbol(old, new, dry_run)` does a word-boundary, file-type-
  whitelisted bulk rename across the workspace (with `dry_run` preview).
* `run_focused_tests(paths)` falls back to git-changed test files when
  no paths are given, so the agent can iterate without re-running the
  whole suite.
* `web_fetch(url)` is HTTPS-only by default, allow-listed via
  `ADA_FETCH_ALLOWLIST`, and reduces HTML to plain text. Set
  `ADA_FETCH_ALLOW_HTTP=1` to permit `http://` for local fixtures.

## Impact analysis & diff stats

* `impact_analysis(files)` walks `tests/` for imports of the changed
  modules and returns just the test files that could be affected.
* `diff_stats(ref)` summarises the pending git diff with per-file
  added/removed counts plus a coarse `safe`/`review`/`risky` verdict.

## Persistent task queue

`task_add`, `task_list`, `task_update`, `task_remove` manage a JSON-
backed queue at `.ada/tasks.json` with statuses
`pending|in_progress|done|blocked` so multi-goal sessions survive
restarts.

## Notebooks

`read_notebook(path)` returns cells (source + a brief output preview)
and `edit_notebook_cell(path, index, source)` rewrites a single cell
without touching outputs/metadata. No `nbformat` dependency.

## Persistent K/V memory

`memory_set/get/list/delete(ns, key, value)` is backed by a SQLite
database at `.ada/memory.sqlite` (WAL mode, thread-local connections
to avoid the macOS Python 3.9 segfault). Use it for cross-run
preferences and lookup caches.

## Profiling, PR drafting, flake hunting

* `profile_run(code)` runs a snippet under cProfile and returns the
  hottest funcs by cumulative time.
* `generate_pr_description(ref)` asks the planner to write a markdown
  PR description from `git log` + `git diff` and saves it to
  `.ada/pr_description.md`.
* `flake_check(paths, runs)` re-runs pytest N times to flag tests that
  flip outcome.

## Context compaction

Long sessions inevitably bloat the chat history (tool results, file dumps,
diffs). Ada auto-summarises the **middle** of the conversation via the
planner model when the live prompt exceeds a token budget, keeping the system
prompt, the kickoff message, and the most recent turns intact.

Tunable via env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ADA_COMPACT_TOKENS` | `24000` | Trigger compaction once total prompt tokens exceed this |
| `ADA_COMPACT_AFTER` | `200` | Safety-net trigger by raw message count |
| `ADA_KEEP_RECENT` | `20` | Tail messages kept verbatim (must cover any in-flight tool-call pairs) |

Token counting uses **tiktoken** (`cl100k_base`) when installed, with a
`len(text)/3` fallback otherwise. The transcript fed to the planner is itself
capped at half the budget so summarisation cannot blow up the planner's window;
the oldest messages in the slice are elided first if needed.

The compacted block is inserted as a single `[CONTEXT SUMMARY OF EARLIER STEPS]`
user message; subsequent compactions fold the previous summary into the new
one rather than stacking.

## Provider profiles

Switch model providers in one flag — Ada ships with built-in presets that set
`OPENAI_BASE_URL`, the API-key env var, and worker/planner defaults. Explicit
env vars and CLI flags still override.

```bash
python main.py . --profile openai      # gpt-4o-mini / gpt-4o
python main.py . --profile deepseek    # deepseek-chat / deepseek-reasoner
python main.py . --profile anthropic   # claude-sonnet-4-5 / claude-opus-4-5
python main.py . --profile moonshot    # moonshot-v1-32k / moonshot-v1-128k
python main.py . --profile ollama      # qwen3.5:9b / qwen3.5:27b (default)
```

The profile reads its API key from a provider-specific env var
(`DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `MOONSHOT_API_KEY`, etc.) so you can
keep multiple keys in one `.env` and swap providers without re-exporting.
Equivalent to setting `ADA_PROFILE=<name>`.

After every run Ada prints a token-usage summary for both worker and planner
(prompt / completion / total / request count), making head-to-head comparisons
between model choices straightforward.

## Try it on the demo project

A tiny buggy package lives in `examples/demo_project/` (5 failing tests).

```bash
cd examples/demo_project && git init -q && git add -A \
  && git -c user.email=demo@x -c user.name=demo commit -q -m "baseline"
cd ../..
python main.py examples/demo_project --goal "让所有 pytest 测试通过，不破坏公共 API"
```

## Safety notes

- Ada **runs shell commands** in your project. Review the `--goal` and watch
  the console; commands are echoed before execution.
- Recommend running on a fresh git branch so you can `git diff` / revert.
- Set tight `--cmd-timeout` when targeting unfamiliar code.

## Project layout

```
.
├── prompt.md            Ada's behaviour specification (system prompt)
├── main.py              CLI entry point
├── server.py            Flask web server (SSE + REST)
├── requirements.txt
├── .env.example         Configuration template
├── ARCHITECTURE.md      Technical design deep-dive
│
├── ada/
│   ├── __init__.py      Public API: `from ada import Ada`
│   ├── agent.py         Tool-calling loop + Ada class
│   ├── llm.py           LLM client (worker + planner routing)
│   ├── tools.py         18 tool implementations + JSON schemas
│   ├── workspace.py     .ada/ artifact manager + path sandbox
│   └── git_ops.py       Git CLI wrapper
│
├── web/
│   └── index.html       Single-page application
│
└── examples/
    ├── run_demo.sh      Demo helper script
    └── demo_project/    Buggy project for testing Ada
```

---

## License

[MIT](LICENSE) © 2026 Jiangsheng Yu
