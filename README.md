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
ollama pull qwen3-coder        # default worker model
ollama pull qwen3              # default planner model
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

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `$ADA_MODEL` or `gpt-4o-mini` | Override LLM model |
| `--steps` | `80` | Max tool-loop iterations |
| `--cmd-timeout` | `120` | Per-shell-command timeout (s) |

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

You can stop and resume across sessions — Ada re-reads these on next launch.

Two tiers, both default to **local Ollama**:

```bash
export ADA_WORKER_MODEL=qwen3-coder   # drives the tool loop (fast, code-tuned)
export ADA_PLANNER_MODEL=qwen3 p` | Read-only exploration |
| `write_file`, `edit_file` | Targeted changes (sandboxed to target dir) |
| `run_command` | Install / build / test / bench / lint |
| `git_status`, `git_diff`, `git_create_branch`, `git_commit`, `git_revert` | Version-control workflow (auto-branch on launch, commit per verified change) |
| `consult_planner` | Escalate hard planning/design questions to the stronger planner model |
| `update_artifact`, `append_journal`, `append_metric`, `read_artifact` | Maintain `.ada/` artifacts |
| `ask_user` | Pause for human input |
| `finish` | Declare convergence with a final summary |

All file paths are sandboxed inside the target directory; attempts to escape
raise `PermissionError`.

## Multi-model routing

Set two models for cost/quality balance:

```bash
export ADA_WORKER_MODEL=gpt-4o-mini   # drives the tool loop (cheap, fast)
export ADA_PLANNER_MODEL=gpt-4o       # one-shot advisor on hard questions
```

Or pass `--model` and `--planner` on the CLI. If only `ADA_MODEL` is set,
both tiers fall back to it.

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
├── prompt.md           # Ada's behaviour spec (system prompt)
├── main.py             # CLI entry
├── ada/
│   ├── agent.py        # main loop
│   ├── llm.py          # OpenAI-compatible client
│   ├── tools.py        # tool implementations + JSON schemas
│   └── workspace.py    # .ada/ artifact manager + path sandbox
├── requirements.txt
└── .env.example
```
