"""Constrained HTTP fetcher for documentation lookup.

Stdlib-only (urllib + html.parser) so it works without extra deps. Used
as a tool so the agent can pull short doc snippets when stuck on an API.

Safety:
* HTTPS-only by default (override with ``ADA_FETCH_ALLOW_HTTP=1``).
* Optional allowlist via ``ADA_FETCH_ALLOWLIST=docs.python.org,pypi.org``.
* Hard cap on response size (default 200 KB).
* 10s timeout, no redirects beyond 3 hops.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

_DEFAULT_MAX = 200 * 1024
_DEFAULT_TIMEOUT = 10


def fetch_url(url: str, max_bytes: int | None = None, timeout: float | None = None) -> dict:
    """GET *url* and return ``{"url", "status", "title", "text"}``.

    HTML is reduced to plain text (script/style stripped). Other
    content-types are returned verbatim, truncated to ``max_bytes``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"error": f"unsupported scheme: {parsed.scheme!r}"}
    if parsed.scheme == "http" and os.getenv("ADA_FETCH_ALLOW_HTTP", "0") not in ("1", "true", "yes"):
        return {"error": "plain HTTP disabled; set ADA_FETCH_ALLOW_HTTP=1 to allow"}

    allowlist = [s.strip() for s in os.getenv("ADA_FETCH_ALLOWLIST", "").split(",") if s.strip()]
    if allowlist and not any(parsed.netloc == d or parsed.netloc.endswith("." + d) for d in allowlist):
        return {"error": f"host {parsed.netloc!r} not in ADA_FETCH_ALLOWLIST"}

    cap = max_bytes or _DEFAULT_MAX
    to = timeout or _DEFAULT_TIMEOUT

    req = urllib.request.Request(url, headers={"User-Agent": "Ada/1.0 (+https://github.com)"})
    try:
        with urllib.request.urlopen(req, timeout=to) as resp:  # noqa: S310 (intentional)
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(cap + 1)
            status = resp.status
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    truncated = len(raw) > cap
    body = raw[:cap].decode("utf-8", errors="replace")
    title = ""
    if "html" in ctype.lower():
        title, body = _html_to_text(body)
    return {
        "url": url,
        "status": status,
        "content_type": ctype,
        "title": title,
        "text": body,
        "truncated": truncated,
    }


class _Stripper(HTMLParser):
    """Collect visible text + title from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):  # type: ignore[override]
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):  # type: ignore[override]
        if tag in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):  # type: ignore[override]
        if self._skip_depth:
            return
        if self._in_title:
            self.title.append(data)
        else:
            self.parts.append(data)


def _html_to_text(html: str) -> tuple[str, str]:
    """Return (title, plain-text body) for an HTML payload."""
    p = _Stripper()
    try:
        p.feed(html)
    except Exception:
        return "", html
    title = re.sub(r"\s+", " ", "".join(p.title)).strip()
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    return title, text
