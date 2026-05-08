"""Read-only Jupyter notebook (``.ipynb``) helpers.

Notebooks are JSON; we expose just enough to:

* :func:`read_notebook` — return cells as ``[{index, type, source, outputs_brief}]``.
* :func:`edit_cell_source` — replace a single cell's source (preserves
  outputs/metadata) and write the file back.

Cell editing is intentionally line-based and surgical so the agent can
fix bugs in a notebook without us having to model rich output blobs.
"""
from __future__ import annotations

import json
from pathlib import Path

# Cap on output preview characters per cell — notebooks routinely embed
# huge plot data URLs that would blow out the model's context window.
_OUTPUT_PREVIEW = 200


def read_notebook(path: Path) -> dict:
    """Return a flat per-cell summary suitable for LLM consumption."""
    p = Path(path)
    if not p.is_file():
        return {"error": f"not found: {p}"}
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    cells = nb.get("cells", [])
    out: list[dict] = []
    for i, c in enumerate(cells):
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        out.append({
            "index": i,
            "type": c.get("cell_type", "?"),
            "source": src,
            "outputs_brief": _brief_outputs(c.get("outputs", [])),
            "execution_count": c.get("execution_count"),
        })
    return {
        "path": str(p),
        "kernel": nb.get("metadata", {}).get("kernelspec", {}).get("name"),
        "cells": out,
        "cell_count": len(out),
    }


def edit_cell_source(path: Path, index: int, new_source: str) -> dict:
    """Replace cell *index*'s source; preserves type, outputs, metadata."""
    p = Path(path)
    if not p.is_file():
        return {"error": f"not found: {p}"}
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    cells = nb.get("cells", [])
    if not 0 <= index < len(cells):
        return {"error": f"cell index {index} out of range (have {len(cells)})"}
    # nbformat stores source as a list of lines; mimic that for compatibility.
    cells[index]["source"] = new_source.splitlines(keepends=True) or [""]
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return {"path": str(p), "index": index, "bytes": len(new_source)}


def _brief_outputs(outputs: list) -> str:
    """Collapse a cell's outputs into a short ``text/plain`` preview."""
    if not outputs:
        return ""
    parts: list[str] = []
    for o in outputs:
        if "text" in o:  # stream output
            parts.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
        elif "data" in o:
            data = o.get("data", {})
            if "text/plain" in data:
                v = data["text/plain"]
                parts.append("".join(v) if isinstance(v, list) else v)
        elif o.get("ename"):
            parts.append(f"{o['ename']}: {o.get('evalue', '')}")
    blob = "\n".join(parts).strip()
    return blob[:_OUTPUT_PREVIEW] + ("..." if len(blob) > _OUTPUT_PREVIEW else "")
