"""CLI entry point for Ada \u2014 autonomous code-improvement agent.

Usage examples
--------------
# Improve a project using local Ollama (default):
python main.py /path/to/project --goal "\u8ba9\u6240\u6709\u6d4b\u8bd5\u901a\u8fc7\u5e76\u6d88\u9664 lint \u8b66\u544a"

# Use a specific model pair:
python main.py /path/to/project \\
    --model qwen3-coder:30b \\
    --planner qwen3.5:27b \\
    --goal "Refactor the API layer for better error handling"

# Cap iterations and per-command timeout:
python main.py /path/to/project --steps 40 --cmd-timeout 60

# Run on Ada itself (self-improvement):
python main.py . --goal "Add type hints and docstrings throughout the ada/ package"

# Disable the auto-created working branch (commits go to current branch):
python main.py /path/to/project --no-auto-branch

Environment variables (set in .env or export):
  OPENAI_BASE_URL    \u2014 API endpoint  (default: http://localhost:11434/v1)
  OPENAI_API_KEY     \u2014 API key       (default: ollama)
  ADA_WORKER_MODEL   \u2014 worker model  (default: qwen3.5:9b)
  ADA_PLANNER_MODEL  \u2014 planner model (default: qwen3.5:27b)
  ADA_MAX_STEPS      \u2014 loop cap      (default: 80)
  ADA_CMD_TIMEOUT    \u2014 shell timeout (default: 120)
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from ada import Ada


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ada",
        description="Ada \u2014 autonomous code-improvement agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ada reads your project, builds a plan, makes changes, verifies them\n"
            "via tests/lint, commits each verified item, and loops until the goal\n"
            "is reached or max_steps is exhausted.\n\n"
            "All changes are made on an auto-created git branch so your main\n"
            "branch stays clean.  Review with:  git diff main..ada/<ts>"
        ),
    )
    parser.add_argument(
        "target_dir",
        help="Directory containing the code to improve.",
    )
    parser.add_argument(
        "--goal", default="",
        help=(
            "High-level objective for Ada, e.g. "
            "'\u8ba9\u6240\u6709\u6d4b\u8bd5\u901a\u8fc7\u5e76\u6d88\u9664 lint \u8b66\u544a'.  "
            "If omitted, Ada will ask."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        metavar="MODEL",
        help="Worker model name.  Overrides ADA_WORKER_MODEL env var.",
    )
    parser.add_argument(
        "--planner", default=None,
        metavar="MODEL",
        help="Planner model name.  Overrides ADA_PLANNER_MODEL env var.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        metavar="N",
        help="Maximum tool-loop iterations (default: $ADA_MAX_STEPS or 80).",
    )
    parser.add_argument(
        "--cmd-timeout", type=int, default=None,
        metavar="SEC",
        help="Per-shell-command timeout in seconds (default: $ADA_CMD_TIMEOUT or 120).",
    )
    parser.add_argument(
        "--no-auto-branch", action="store_true",
        help="Skip auto-creating an ada/<timestamp> working branch.",
    )
    args = parser.parse_args()

    ada = Ada(
        target_dir=args.target_dir,
        user_goal=args.goal,
        model=args.model,
        planner_model=args.planner,
        max_steps=args.steps,
        cmd_timeout=args.cmd_timeout,
        auto_branch=not args.no_auto_branch,
    )
    ada.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
