"""Lightweight safety filters: dangerous-shell-command guard + secrets scan.

These are advisory, not airtight. Their goal is to reduce the most common
foot-guns the LLM steps on (`rm -rf .`, `git push --force`, accidentally
committing an API key) without getting in the way of legitimate work.

Both filters can be **disabled** by setting ``ADA_SAFETY=0`` for users who
want to opt out (e.g. CI pipelines that do their own sandboxing).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ── dangerous-command guard ──────────────────────────────────────────────


# Each pattern is matched against the FULL stripped command string.
# We keep the list small & high-signal: false positives hurt UX more than
# the small extra protection of a bigger blacklist.
_DANGER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*)\s+(/|~|\$HOME|\.|\*)",
     "rm -rf on root, home, cwd, or wildcard"),
    # Catch rm -rf with command substitution / variable expansion targets.
    (r"\brm\s+-[a-zA-Z]*[rRf][a-zA-Z]*\s+[\"']?\$[\w({]",
     "rm -rf with variable/command-substitution target"),
    (r"\bfind\s+\S+.*-delete\b", "find … -delete"),
    (r"\bfind\s+\S+.*-exec\s+rm\b", "find … -exec rm"),
    (r"\balias\s+\w+\s*=", "shell alias redefinition"),
    (r"\bsudo\b", "sudo escalation"),
    (r"\bchmod\s+-R\s+0?7?77\b", "chmod -R 777"),
    (r"\bchown\s+-R\b", "recursive chown"),
    (r":\s*\(\)\s*\{.*:.*\|.*:.*\}\s*;.*:", "fork bomb"),
    (r">\s*/dev/sd[a-z]\b", "writing directly to disk device"),
    (r"\bmkfs(\.|\b)", "filesystem format"),
    (r"\bdd\s+if=.*of=/dev/", "dd to device"),
    (r"\bgit\s+push\s+(.*\s)?(--force|-f)\b", "git push --force"),
    (r"\bgit\s+reset\s+--hard\b.*(origin|HEAD~|main|master)",
     "git reset --hard on tracked ref"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*[fdx]", "git clean -fdx"),
    (r"\bcurl\s+[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b", "curl … | sh"),
    (r"\bwget\s+[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b", "wget … | sh"),
    (r"\beval\s+\$\(curl\b", "eval $(curl …)"),
    (r"\bnpm\s+publish\b", "npm publish"),
    (r"\bpip\s+(install|uninstall).*--break-system-packages", "system pip override"),
    (r"\bdocker\s+system\s+prune\s+-a", "docker system prune -a"),
    (r"\bkubectl\s+delete\s+(ns|namespace|cluster)", "kubectl delete cluster/ns"),
    (r"(?i)\bdrop\s+(table|database|schema)\b", "SQL DROP statement"),
    (r"(?i)\btruncate\s+table\b", "SQL TRUNCATE TABLE"),
)


@dataclass(frozen=True)
class CommandRisk:
    is_dangerous: bool
    reason: str = ""

    def to_error(self) -> dict:
        return {
            "blocked": True,
            "reason": self.reason,
            "hint": (
                "If you really need to run this, ask the human via `ask_user` "
                "for confirmation; or set ADA_SAFETY=0 to disable the guard."
            ),
        }


def _safety_enabled() -> bool:
    return os.getenv("ADA_SAFETY", "1").lower() not in ("0", "false", "no", "off")


def assess_command(cmd: str) -> CommandRisk:
    """Return a non-dangerous result when the guard is disabled or no rule fires."""
    if not _safety_enabled() or not cmd:
        return CommandRisk(False)
    s = cmd.strip()
    for pattern, label in _DANGER_PATTERNS:
        if re.search(pattern, s):
            return CommandRisk(True, label)
    return CommandRisk(False)


# ── secrets scan ─────────────────────────────────────────────────────────


# Patterns chosen for high precision. Each name → compiled regex.
# Generic high-entropy detection is intentionally NOT included because it
# fires on commit hashes, base64 fixtures, etc.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key-id",     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws-secret-access-key",
        re.compile(r"(?i)aws.{0,20}?(secret|sk).{0,20}?['\"][A-Za-z0-9/+=]{40}['\"]")),
    ("github-token",          re.compile(r"\bghp_[A-Za-z0-9]{36,}\b")),
    ("github-fine-grained",   re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82,}\b")),
    ("openai-key",            re.compile(r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b")),
    ("openai-project-key",    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{40,}\b")),
    ("anthropic-key",         re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{30,}\b")),
    ("google-api-key",        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack-token",           re.compile(r"\bxox[pboar]-[A-Za-z0-9-]{10,}\b")),
    ("stripe-key",            re.compile(r"\b(sk|rk)_(live|test)_[A-Za-z0-9]{24,}\b")),
    ("private-key-block",     re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----")),
    ("jwt",                   re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
)


@dataclass(frozen=True)
class SecretHit:
    kind: str
    line: int
    snippet: str


def scan_secrets(text: str, max_hits: int = 10) -> list[SecretHit]:
    """Return likely credential leaks in *text* (line-numbered, 1-based)."""
    if not _safety_enabled() or not text:
        return []
    hits: list[SecretHit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pat in _SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                snip = line.strip()
                if len(snip) > 120:
                    snip = snip[:117] + "..."
                hits.append(SecretHit(kind, lineno, snip))
                if len(hits) >= max_hits:
                    return hits
                break  # don't double-count the same line
    return hits
