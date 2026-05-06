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
| `run_command` | Shell | Execute a shell command; output tail-truncated to 8 KB |
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
