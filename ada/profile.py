"""cProfile-based hot-spot reporting for Ada.

:func:`profile_python` runs a short Python snippet under cProfile and
returns the top-N functions by cumulative time.  The snippet runs in
the current process — no subprocess hop — so we can re-use the agent's
already-loaded modules.
"""
from __future__ import annotations

import cProfile
import io
import pstats


def profile_python(code: str, top: int = 15, sort: str = "cumulative") -> dict:
    """Profile a snippet of Python; return a list of hot functions.

    The snippet runs with a minimal globals dict.  Exceptions surface
    as ``{"error": ...}`` rather than propagating, since the agent must
    be able to recover.
    """
    if top <= 0:
        return {"error": "top must be positive"}
    pr = cProfile.Profile()
    try:
        pr.enable()
        exec(compile(code, "<ada-profile>", "exec"), {"__name__": "__profile__"})
        pr.disable()
    except Exception as exc:  # noqa: BLE001 — surface arbitrary user errors
        pr.disable()
        return {"error": f"{type(exc).__name__}: {exc}"}

    stats = pstats.Stats(pr).sort_stats(sort)
    rows: list[dict] = []
    # ``stats.stats`` is a dict[(file, line, func)] -> (cc, nc, tt, ct, callers).
    # We pull the same data the human-friendly formatter uses but in
    # structured form so the agent can reason about it.
    for (fn, lineno, func), (cc, nc, tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append({
            "func": f"{fn}:{lineno}({func})",
            "calls": nc,
            "primitive_calls": cc,
            "tottime": round(tt, 6),
            "cumtime": round(ct, 6),
        })
    rows.sort(key=lambda r: r["cumtime" if sort == "cumulative" else "tottime"], reverse=True)
    return {"top": rows[:top], "total_funcs": len(rows)}
