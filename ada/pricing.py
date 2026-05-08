"""USD cost estimator for LLM token usage.

Prices are expressed in USD per **million tokens** (USD/M).  We do prefix
matching on the model name so e.g. ``gpt-4o-2024-11-20`` matches the
``gpt-4o`` entry without needing per-version rows.

The numbers are reasonable as of the package's last refresh, but they are
**estimates** — for billing-grade accounting consult your provider invoice.
Local / Ollama models price at $0.

You can override or extend the table at runtime via :py:func:`set_price`.
"""
from __future__ import annotations


# (input_per_million, output_per_million)  — USD
_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini":          (0.15, 0.60),
    "gpt-4o":               (2.50, 10.00),
    "gpt-4.1-mini":         (0.40, 1.60),
    "gpt-4.1":              (2.00, 8.00),
    "o1-mini":              (1.10, 4.40),
    "o1":                   (15.00, 60.00),
    "o3-mini":              (1.10, 4.40),
    "o3":                   (10.00, 40.00),

    # Anthropic
    "claude-3-5-haiku":     (0.80, 4.00),
    "claude-3-5-sonnet":    (3.00, 15.00),
    "claude-3-7-sonnet":    (3.00, 15.00),
    "claude-opus-4":        (15.00, 75.00),
    "claude-sonnet-4":      (3.00, 15.00),

    # DeepSeek
    "deepseek-chat":        (0.27, 1.10),
    "deepseek-reasoner":    (0.55, 2.19),

    # Moonshot
    "kimi":                 (1.00, 1.00),

    # Qwen / cloud
    "qwen-max":             (2.40, 9.60),
    "qwen-plus":            (0.40, 1.20),
    "qwen-turbo":           (0.05, 0.20),
}


def set_price(model_prefix: str, input_per_m: float, output_per_m: float) -> None:
    """Add/override a model's price table at runtime."""
    _PRICES[model_prefix] = (float(input_per_m), float(output_per_m))


def _lookup(model: str) -> tuple[float, float] | None:
    if not model:
        return None
    name = model.lower()
    # local backends are free
    for prefix in ("ollama/", "ollama:", "llama", "qwen3.5", "qwen3-coder",
                   "qwen2.5", "phi", "mistral:", "mixtral:", "gemma:"):
        if name.startswith(prefix) or f"/{prefix}" in name:
            return (0.0, 0.0)
    # exact match
    if model in _PRICES:
        return _PRICES[model]
    # longest prefix wins
    matches = [k for k in _PRICES if model.startswith(k)]
    if matches:
        return _PRICES[max(matches, key=len)]
    return None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """Return ``{"usd": float|None, "model": str, "priced": bool}``.

    ``usd`` is ``None`` when the model is unknown — the caller should treat
    that as "free / unmetered" rather than zero, so the UI can distinguish.
    """
    p = _lookup(model)
    if p is None:
        return {"model": model, "priced": False, "usd": None}
    inp, out = p
    usd = (prompt_tokens / 1_000_000) * inp + (completion_tokens / 1_000_000) * out
    return {"model": model, "priced": True, "usd": round(usd, 6)}
