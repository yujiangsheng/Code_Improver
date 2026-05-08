"""Embedding-based semantic code search.

A lightweight vector index for the project, persisted to
``.ada/embeddings.db`` (SQLite + raw float32 BLOB vectors).  Powers the
``semantic_search`` tool — finds *conceptually* related code even when the
literal keywords don't appear (e.g. "retry logic" → finds a function that
loops over ``backoff = 2 ** attempt`` without using the word "retry").

Design
------
* **Optional dependency** — needs an embeddings-capable endpoint.  If
  ``OPENAI_API_KEY`` is unset and ``ADA_EMBED_FAKE`` is also unset, the
  index reports ``available() == False`` and the tool returns a structured
  ``{"error": "embeddings not configured"}``.
* **Incremental** — files are skipped on re-index when ``(mtime, size)``
  matches the recorded values, so day-to-day repo evolution costs only the
  delta.
* **No heavy deps** — pure-Python cosine search; numpy used only if already
  installed.  Vectors are stored as raw little-endian float32 to keep the
  DB tight.
* **Symbol-aware chunking** — when tree-sitter is available, chunks follow
  function/class boundaries (better recall).  Falls back to overlapping
  line windows.

Configuration (env)
-------------------
* ``ADA_EMBED_MODEL``    — OpenAI embeddings model (default
                            ``text-embedding-3-small``, dim 1536).
* ``ADA_EMBED_BASE_URL`` — Override endpoint; defaults to
                            ``OPENAI_BASE_URL`` then OpenAI public.
* ``ADA_EMBED_API_KEY``  — Override key; defaults to ``OPENAI_API_KEY``.
* ``ADA_EMBED_FAKE``     — If ``1``, use a deterministic hash-based fake
                            embedder (offline-friendly, used by tests).
* ``ADA_EMBED_BATCH``    — Embedding batch size (default 64).
* ``ADA_EMBED_MAX_BYTES``— Skip files larger than this (default 200 000).
"""
from __future__ import annotations

import array
import hashlib
import math
import os
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable

# Files / directories to ignore during walk.
_SKIP_DIRS = {
    ".git", ".ada", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
    ".idea", ".vscode", ".next", ".cache",
}
# Extensions worth indexing (text + code).  Anything else is skipped.
_TEXT_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".scala", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".swift",
    ".sh", ".bash", ".zsh", ".sql",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".json", ".ini",
}

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIM = 1536  # for the fake embedder; real embedder uses model's dim

# ── chunking parameters ──────────────────────────────────────────────────
# Window size and overlap used when symbol-aware chunking isn't available.
_LINE_WINDOW = 40
_LINE_OVERLAP = 10
_MAX_CHUNK_CHARS = 4000  # hard cap per chunk — embedding APIs limit input length


# ─── embedder backends ──────────────────────────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Deterministic hash-based embedder used for offline tests.

    Maps each input string to a unit-norm vector via repeated SHA-256
    hashing.  Two identical inputs map to the same vector, two similar
    inputs map to *different* vectors — so it cannot test recall quality,
    only the storage/retrieval plumbing.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.dim = dim
        self.model = "fake-hash"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec: list[float] = []
            seed = t.encode("utf-8")
            counter = 0
            while len(vec) < self.dim:
                h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
                # 8 floats per 32-byte hash (4 bytes each, mapped to [-1, 1]).
                for i in range(0, 32, 4):
                    n = int.from_bytes(h[i:i + 4], "big", signed=True)
                    vec.append(n / 2_147_483_647.0)
                    if len(vec) >= self.dim:
                        break
                counter += 1
            out.append(_normalize(vec))
        return out


class _OpenAIEmbedder:
    """Wrapper around any OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # local import — keeps module import cheap
        self.model = model or os.getenv("ADA_EMBED_MODEL") or _DEFAULT_MODEL
        self._client = OpenAI(
            api_key=(
                api_key
                or os.getenv("ADA_EMBED_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            ),
            base_url=(
                base_url
                or os.getenv("ADA_EMBED_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            ),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [_normalize(list(d.embedding)) for d in resp.data]


def _make_embedder() -> Any | None:
    """Build an embedder per env config, or return ``None`` if unconfigured."""
    if os.getenv("ADA_EMBED_FAKE", "").lower() in ("1", "true", "yes", "on"):
        return _FakeEmbedder()
    api_key = os.getenv("ADA_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return _OpenAIEmbedder()
    except Exception:
        return None


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector so cosine similarity reduces to a dot product."""
    s = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / s for x in vec]


def _pack(vec: list[float]) -> bytes:
    """Encode a float vector as raw little-endian float32 bytes."""
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    """Decode a raw little-endian float32 byte blob into a Python list."""
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


# ─── chunking ───────────────────────────────────────────────────────────────────────────────────────────────


def _line_chunks(text: str) -> list[tuple[int, int, str]]:
    """Split *text* into overlapping line-windows.

    Returns ``[(start_line_1based, end_line_1based, snippet)]``.  Used as
    the universal fallback when no language-aware splitter applies.
    """
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[int, int, str]] = []
    n = len(lines)
    step = max(1, _LINE_WINDOW - _LINE_OVERLAP)
    i = 0
    while i < n:
        end = min(n, i + _LINE_WINDOW)
        snippet = "\n".join(lines[i:end])
        if len(snippet) > _MAX_CHUNK_CHARS:
            snippet = snippet[:_MAX_CHUNK_CHARS]
        chunks.append((i + 1, end, snippet))
        if end >= n:
            break
        i += step
    return chunks


def _symbol_chunks(
    root: Path, path: Path, text: str
) -> list[tuple[int, int, str]] | None:
    """Try to chunk *path*'s text along symbol boundaries (functions/classes).

    Returns ``None`` if tree-sitter isn't available or the language has no
    parser, signalling the caller to fall back to ``_line_chunks``.
    """
    try:
        from .semantic import Semantic
    except Exception:
        return None
    sem = Semantic(root)
    if not sem.available():
        return None
    try:
        syms = sem.list_symbols(str(path.relative_to(root)))
    except Exception:
        return None
    if not syms:
        return None
    lines = text.splitlines()
    n = len(lines)
    chunks: list[tuple[int, int, str]] = []
    for s in syms:
        # ``Symbol`` is a dataclass; access via attributes (not .get).
        start = max(1, int(getattr(s, "start_line", 1) or 1))
        end = min(n, int(getattr(s, "end_line", start) or start))
        if end < start:
            continue
        snippet = "\n".join(lines[start - 1:end])
        if not snippet.strip():
            continue
        if len(snippet) > _MAX_CHUNK_CHARS:
            snippet = snippet[:_MAX_CHUNK_CHARS]
        chunks.append((start, end, snippet))
    return chunks or None


def _chunk_file(root: Path, path: Path) -> list[tuple[int, int, str]]:
    """Produce ``[(start, end, text)]`` chunks for a single file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text.strip():
        return []
    sym = _symbol_chunks(root, path, text)
    if sym:
        return sym
    return _line_chunks(text)


# ─── index ──────────────────────────────────────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    vector      BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

CREATE TABLE IF NOT EXISTS files (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class EmbedIndex:
    """Persistent embedding index for one repo.

    Use :py:meth:`index` to (re)build, :py:meth:`search` to query, and
    :py:meth:`available` to detect whether the embedder is wired up.
    """

    def __init__(
        self,
        root: str | Path,
        db_path: str | Path | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        ada_dir = self.root / ".ada"
        ada_dir.mkdir(exist_ok=True)
        self.db_path = Path(db_path) if db_path else ada_dir / "embeddings.db"
        self._embedder = embedder if embedder is not None else _make_embedder()
        self._max_bytes = int(os.getenv("ADA_EMBED_MAX_BYTES", "200000"))
        self._batch = max(1, int(os.getenv("ADA_EMBED_BATCH", "64")))
        self._init_db()

    # ── public API ──────────────────────────────────────────────────────

    def available(self) -> bool:
        """Return ``True`` iff an embedder backend is configured."""
        return self._embedder is not None

    def stats(self) -> dict[str, Any]:
        """Return ``{chunks, files, model, db_path}`` for the live index."""
        with self._connect() as conn:
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {
            "chunks": chunks,
            "files": files,
            "model": getattr(self._embedder, "model", None),
            "db_path": str(self.db_path),
        }

    def clear(self) -> None:
        """Drop all chunks and file records (keeps schema)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM files")

    def index(
        self,
        paths: Iterable[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Embed and store chunks for the given files (default: whole repo).

        Parameters
        ----------
        paths:
            Iterable of repo-relative paths.  If ``None``, walk the tree.
        force:
            If ``True``, re-embed even if ``(mtime, size)`` is unchanged.

        Returns
        -------
        dict
            ``{"indexed", "skipped", "deleted", "errors", "chunks"}``.
        """
        if not self.available():
            return {"error": "embeddings not configured (set OPENAI_API_KEY or ADA_EMBED_FAKE=1)"}

        targets = list(self._iter_files(paths))
        indexed = skipped = chunks_added = errors = 0
        with self._connect() as conn:
            existing = {
                row[0]: (row[1], row[2], row[3])
                for row in conn.execute(
                    "SELECT path, mtime, size, model FROM files"
                )
            }
            seen: set[str] = set()
            current_model = getattr(self._embedder, "model", "?")

            pending_text: list[str] = []
            pending_meta: list[tuple[str, int, int, str]] = []  # (rel, start, end, snippet)
            removed_paths: list[str] = []

            def _flush() -> int:
                """Embed and insert any buffered chunks. Returns number written."""
                if not pending_text:
                    return 0
                vecs = self._embedder.embed(pending_text)
                rows = [
                    (rel, start, end, snippet, _pack(vec))
                    for (rel, start, end, snippet), vec in zip(pending_meta, vecs)
                ]
                conn.executemany(
                    "INSERT INTO chunks(path,start_line,end_line,text,vector) "
                    "VALUES (?,?,?,?,?)",
                    rows,
                )
                n = len(rows)
                pending_text.clear()
                pending_meta.clear()
                return n

            for abspath in targets:
                rel = str(abspath.relative_to(self.root))
                seen.add(rel)
                try:
                    st = abspath.stat()
                except OSError:
                    errors += 1
                    continue
                if st.st_size > self._max_bytes:
                    skipped += 1
                    continue
                prior = existing.get(rel)
                if (
                    not force
                    and prior is not None
                    and prior[0] == st.st_mtime
                    and prior[1] == st.st_size
                    and prior[2] == current_model
                ):
                    skipped += 1
                    continue

                file_chunks = _chunk_file(self.root, abspath)
                if not file_chunks:
                    skipped += 1
                    continue

                # Remove old rows for this file before inserting new ones.
                conn.execute("DELETE FROM chunks WHERE path = ?", (rel,))
                for start, end, snippet in file_chunks:
                    pending_text.append(snippet)
                    pending_meta.append((rel, start, end, snippet))
                    if len(pending_text) >= self._batch:
                        try:
                            chunks_added += _flush()
                        except Exception:
                            errors += 1
                            pending_text.clear()
                            pending_meta.clear()

                conn.execute(
                    "INSERT OR REPLACE INTO files(path,mtime,size,model) VALUES (?,?,?,?)",
                    (rel, st.st_mtime, st.st_size, current_model),
                )
                indexed += 1

            try:
                chunks_added += _flush()
            except Exception:
                errors += 1
                pending_text.clear()
                pending_meta.clear()

            # Garbage-collect rows for deleted files (only when scanning the
            # whole tree; partial reindex must not delete unrelated entries).
            if paths is None:
                for rel in list(existing.keys()):
                    if rel not in seen and not (self.root / rel).exists():
                        conn.execute("DELETE FROM chunks WHERE path = ?", (rel,))
                        conn.execute("DELETE FROM files WHERE path = ?", (rel,))
                        removed_paths.append(rel)

        return {
            "indexed": indexed,
            "skipped": skipped,
            "deleted": len(removed_paths),
            "errors": errors,
            "chunks_added": chunks_added,
            "model": getattr(self._embedder, "model", None),
        }

    def search(
        self,
        query: str,
        k: int = 8,
        path_glob: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-*k* most similar chunks to *query*.

        Parameters
        ----------
        query:
            Free-form natural-language or code snippet.
        k:
            Maximum number of hits.
        path_glob:
            Optional ``fnmatch`` pattern (e.g. ``"src/**.py"``) restricting
            the search to matching paths.
        """
        if not self.available():
            return []
        qvec = self._embedder.embed([query])[0]
        with self._connect() as conn:
            if path_glob:
                # SQLite's GLOB syntax differs slightly from fnmatch; we filter
                # in Python after fetching.  Reasonable for repos < 50k chunks.
                rows = conn.execute(
                    "SELECT path, start_line, end_line, text, vector FROM chunks"
                ).fetchall()
                import fnmatch
                rows = [r for r in rows if fnmatch.fnmatch(r[0], path_glob)]
            else:
                rows = conn.execute(
                    "SELECT path, start_line, end_line, text, vector FROM chunks"
                ).fetchall()

        scored: list[tuple[float, tuple]] = []
        for row in rows:
            vec = _unpack(row[4])
            scored.append((_dot(qvec, vec), row))
        scored.sort(key=lambda t: t[0], reverse=True)

        out: list[dict[str, Any]] = []
        for score, row in scored[:k]:
            text = row[3]
            preview = text if len(text) <= 800 else text[:800] + "\n... [truncated]"
            out.append({
                "path": row[0],
                "start_line": row[1],
                "end_line": row[2],
                "score": round(score, 4),
                "snippet": preview,
            })
        return out

    # ── internals ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection (auto-commit on context exit)."""
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        """Create tables if missing."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _iter_files(self, paths: Iterable[str] | None) -> Iterable[Path]:
        """Yield absolute paths to candidate files for indexing."""
        if paths is not None:
            for p in paths:
                ap = (self.root / p).resolve()
                if ap.is_file() and ap.suffix in _TEXT_EXT:
                    yield ap
            return
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in _TEXT_EXT:
                    yield p
