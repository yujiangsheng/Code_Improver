"""CLI entry point.

Usage:
    python main.py <target_dir> [--goal "..."] [--model gpt-4o-mini] [--steps 80]
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from ada import Ada


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ada — autonomous code-improvement agent")
    parser.add_argument("target_dir", help="Directory containing the code to improve")
    parser.add_argument("--goal", default="", help="High-level user goal for Ada")
    parser.add_argument("--model", default=None, help="Worker model (overrides ADA_WORKER_MODEL/ADA_MODEL)")
    parser.add_argument("--planner", default=None, help="Planner model (overrides ADA_PLANNER_MODEL)")
    parser.add_argument("--steps", type=int, default=None, help="Override ADA_MAX_STEPS")
    parser.add_argument("--cmd-timeout", type=int, default=None,
                        help="Per-command timeout in seconds")
    parser.add_argument("--no-auto-branch", action="store_true",
                        help="Skip auto-creating an ada/<ts> working branch")
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
