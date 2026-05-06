#!/usr/bin/env bash
# run_demo.sh — bootstrap the demo project and run Ada against it.
#
# Usage:
#   bash examples/run_demo.sh                  # default (qwen3.5:9b worker)
#   bash examples/run_demo.sh --model qwen3-coder:30b
#
# All extra arguments are forwarded to main.py.
#
# Requirements:
#   • Python venv activated (or requirements.txt installed system-wide)
#   • Ollama running with at least qwen3.5:9b pulled
#
# The script is safe to re-run: it skips git init if already a repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="$REPO_ROOT/examples/demo_project"

# ── 1. Initialise git repo inside demo_project if not already done ───────────
if [ ! -d "$DEMO_DIR/.git" ]; then
    echo ">>> Initialising git repo in $DEMO_DIR"
    git -C "$DEMO_DIR" init -q
    git -C "$DEMO_DIR" \
        -c user.email="demo@ada.local" \
        -c user.name="Demo" \
        add -A
    git -C "$DEMO_DIR" \
        -c user.email="demo@ada.local" \
        -c user.name="Demo" \
        commit -q -m "baseline: intentionally buggy stats library"
    echo ">>> Baseline commit created."
else
    echo ">>> Git repo already initialised — skipping init."
fi

# ── 2. Show baseline test results ────────────────────────────────────────────
echo ""
echo ">>> Baseline test results (expected: ~5 failures):"
cd "$DEMO_DIR"
python -m pytest test_stats.py -v --tb=no -q 2>&1 || true
cd "$REPO_ROOT"

# ── 3. Run Ada ────────────────────────────────────────────────────────────────
echo ""
echo ">>> Launching Ada …"
echo "    Target : $DEMO_DIR"
echo "    Goal   : 让所有 pytest 测试通过，不破坏公共 API"
echo ""

python "$REPO_ROOT/main.py" "$DEMO_DIR" \
    --goal "让所有 pytest 测试通过，不破坏公共 API。修复 stats.py 中的 5 个已知 bug；不要改动测试文件。" \
    "$@"

# ── 4. Show final test results ────────────────────────────────────────────────
echo ""
echo ">>> Final test results:"
cd "$DEMO_DIR"
python -m pytest test_stats.py -v 2>&1 || true
