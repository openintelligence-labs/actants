from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from actants.cache.request import KEY_VERSION, CacheRequest
from actants.llm.base import ChatMessage, CompletionResult

if TYPE_CHECKING:
    import sqlite3

    from actants.cache.embeddings import Embedder

log = structlog.get_logger(__name__)

#: On-disk schema version, stored in ``PRAGMA user_version``.
#:
#: Bumped whenever the table layout *or* the meaning of ``scope_hash`` changes — the
#: latter is why it is derived from :data:`~actants.cache.request.KEY_VERSION` rather
#: than being an independent counter. A file written by an actants whose scope hash
#: covered fewer fields must not be read by this one, because its entries were keyed on
#: a subset of the request and would be served for requests they never matched.
SCHEMA_VERSION = KEY_VERSION


def _serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _fingerprint(messages: list[ChatMessage]) -> str:
    """Flatten messages into a single text string for embedding."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    return "\n".join(parts)


class CacheSchemaMismatch(RuntimeError):
    """Raised when an on-disk cache was written by an incompatible schema version.

    Only raised when the cache was constructed with ``on_schema_mismatch="error"``; the
    default is to discard the stale file and start empty.
    """


class SqliteVecCache:
    """Semantic cache backed by sqlite-vec.

    A cache hit requires two things:

    1. an exact match on the request's **scope hash** — provider, model, temperature,
       ``max_tokens``, tool definitions, response format, and the role sequence of the
       conversation; and
    2. a query vector whose cosine distance to a stored vector is below
       ``similarity_threshold``.

    A threshold of 0.05 means roughly "within 95% cosine similarity" — tune per use case.

    Only the message *content* is matched semantically. Everything else is matched
    exactly, because a smaller ``max_tokens`` or a different tool set produces a
    genuinely different answer no matter how similar the prompt looks.

    **On-disk compatibility.** The database records its schema version in
    ``PRAGMA user_version``. Opening a file written by an incompatible version discards
    it and starts empty rather than serving entries that were keyed on fewer fields;
    pass ``on_schema_mismatch="error"`` to raise :class:`CacheSchemaMismatch` instead.
    A cache is disposable, so dropping it costs a re-computation; reading it wrong costs
    a wrong answer.
    """

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        *,
        similarity_threshold: float = 0.05,
        default_ttl: int | None = 3600,
        on_schema_mismatch: str = "reset",
    ) -> None:
        try:
            import sqlite3

            import sqlite_vec
        except ImportError as exc:
            raise ImportError(
                "Install with `pip install actants[cache]` to use SqliteVecCache"
            ) from exc
        if on_schema_mismatch not in ("reset", "error"):
            raise ValueError(
                f"on_schema_mismatch must be 'reset' or 'error', got {on_schema_mismatch!r}. "
                "'reset' discards an incompatible cache file and starts empty; "
                "'error' raises CacheSchemaMismatch so you can handle it yourself."
            )
        self._sqlite3 = sqlite3
        self.path = str(path)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.on_schema_mismatch = on_schema_mismatch
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None
        self._sqlite_vec = sqlite_vec
        self._dim: int | None = None

    def _connect(self, dim: int) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = self._sqlite3.connect(self.path)
        conn.enable_load_extension(True)
        self._sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        self._apply_schema_version(conn)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                temperature REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_scope ON entries (scope_hash)")
        # ``scope_hash`` is a sqlite-vec PARTITION KEY, not an ordinary column on
        # ``entries`` that we filter after the fact. ``MATCH ... AND k = 1`` is a KNN
        # query: it picks the k nearest vectors *first* and only then applies the rest of
        # the WHERE clause. Filtering by scope afterwards therefore turns "the nearest
        # vector in another scope" into a spurious miss — the correct entry is never
        # considered. A partition key prunes to the matching scope before the search.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0("
            f"entry_id INTEGER PRIMARY KEY, scope_hash TEXT PARTITION KEY, "
            f"embedding FLOAT[{dim}])"
        )
        conn.commit()
        self._conn = conn
        self._dim = dim
        return conn

    def _apply_schema_version(self, conn: sqlite3.Connection) -> None:
        """Check ``PRAGMA user_version`` and reset the file if it is incompatible.

        A fresh database reports version 0 and has no tables; that is not a mismatch,
        it is a new file, so it is simply stamped with the current version.
        """
        found = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'entries'"
            ).fetchone()[0]
            > 0
        )

        if found == SCHEMA_VERSION and has_tables:
            return
        if not has_tables:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            return

        if self.on_schema_mismatch == "error":
            raise CacheSchemaMismatch(
                f"Semantic cache at {self.path!r} was written by a different actants "
                f"cache schema (found version {found}, this build expects {SCHEMA_VERSION}). "
                "Its entries were keyed on a different set of request fields, so serving "
                "them now could return an answer generated under different parameters. "
                f"Delete the file to start fresh, or construct the cache with "
                "SqliteVecCache(..., on_schema_mismatch='reset') to have actants do it."
            )

        log.warning(
            "semantic_cache_schema_reset",
            path=self.path,
            found_version=found,
            expected_version=SCHEMA_VERSION,
            reason="entries were keyed on a different set of request fields",
        )
        conn.execute("DROP TABLE IF EXISTS entries")
        conn.execute("DROP TABLE IF EXISTS vec_index")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    async def get_request(self, request: CacheRequest) -> CompletionResult | None:
        """Return a cached result for ``request``, or None.

        Matches ``request.scope_hash()`` exactly and the message embedding by distance.
        """
        embedding = await self.embedder.embed(request.embedding_text())
        return await self._lookup(embedding, request)

    async def set_request(
        self,
        request: CacheRequest,
        value: CompletionResult,
        ttl: int | None = None,
    ) -> None:
        """Store ``value`` as the answer to ``request``."""
        text = request.embedding_text()
        embedding = await self.embedder.embed(text)
        await self._insert(text, embedding, request, value, ttl)

    async def get(self, key: str) -> CompletionResult | None:  # pragma: no cover - protocol impl
        return None

    async def set(
        self,
        key: str,
        value: CompletionResult,
        ttl: int | None = None,
    ) -> None:  # pragma: no cover - protocol impl
        return None

    async def clear(self) -> None:
        async with self._lock:
            if self._conn is None:
                return
            self._conn.execute("DELETE FROM entries")
            self._conn.execute("DELETE FROM vec_index")
            self._conn.commit()

    async def _lookup(
        self,
        embedding: list[float],
        request: CacheRequest,
    ) -> CompletionResult | None:
        async with self._lock:
            conn = self._connect(len(embedding))
            cur = conn.execute(
                """
                SELECT e.id, e.result_json, e.expires_at, v.distance
                FROM vec_index v JOIN entries e ON e.id = v.entry_id
                WHERE v.scope_hash = ?
                  AND v.embedding MATCH ?
                  AND k = 1
                ORDER BY v.distance
                LIMIT 1
                """,
                (request.scope_hash(), _serialize_vector(embedding)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            entry_id, result_json, expires_at, distance = row
            if expires_at is not None and time.time() > expires_at:
                conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
                conn.execute("DELETE FROM vec_index WHERE entry_id = ?", (entry_id,))
                conn.commit()
                return None
            if distance > self.similarity_threshold:
                return None
            return CompletionResult.model_validate(json.loads(result_json))

    async def _insert(
        self,
        fingerprint: str,
        embedding: list[float],
        request: CacheRequest,
        value: CompletionResult,
        ttl: int | None,
    ) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        now = time.time()
        expires_at: float | None = now + effective_ttl if effective_ttl is not None else None
        async with self._lock:
            conn = self._connect(len(embedding))
            cur = conn.execute(
                "INSERT INTO entries (scope_hash, model, temperature, fingerprint, "
                "result_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    request.scope_hash(),
                    request.model,
                    round(request.temperature, 6),
                    fingerprint,
                    value.model_dump_json(),
                    now,
                    expires_at,
                ),
            )
            entry_id = cur.lastrowid
            conn.execute(
                "INSERT INTO vec_index (entry_id, scope_hash, embedding) VALUES (?, ?, ?)",
                (entry_id, request.scope_hash(), _serialize_vector(embedding)),
            )
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __len__(self) -> int:
        if self._conn is None:
            return 0
        count: int = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        return count

    def __repr__(self) -> str:
        return f"SqliteVecCache(path={self.path!r}, threshold={self.similarity_threshold})"

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "threshold": self.similarity_threshold,
            "dim": self._dim,
            "entries": len(self),
            "schema_version": SCHEMA_VERSION,
        }
