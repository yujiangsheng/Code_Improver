"""Ada — autonomous code-improvement agent.

Public API
----------
from ada import Ada

ada = Ada(
    target_dir="/path/to/project",
    user_goal="让所有测试通过并消除 lint 警告",
    model="qwen3.5:9b",        # worker (fast, drives the tool loop)
    planner_model="qwen3.5:27b", # planner (strong, one-shot advisor)
    max_steps=80,
    cmd_timeout=120,
    auto_branch=True,          # creates ada/<ts> branch automatically
)
summary = ada.run()            # blocks until finish or max_steps
"""
from .agent import Ada

__version__ = "0.1.0"
__all__ = ["Ada"]
