from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_kit.llm.base import ChatMessage, CompletionResult

if TYPE_CHECKING:
    import sqlite3

    from agentic_kit.cache.embeddings import Embedder


def _serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _fingerprint(messages: list[ChatMessage]) -> str:
    """Flatten messages into a single text string for embedding."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    return "\n".join(parts)


class SqliteVecCache:
    """Semantic cache backed by sqlite-vec.

    A cache hit requires two things: (1) same provider/model/temperature and (2) a query
    vector whose cosine distance to a stored vector is below ``similarity_threshold``. A
    threshold of 0.05 means roughly "within 95% cosine similarity" — tune per use case.
    """

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        *,
        similarity_threshold: float = 0.05,
        default_ttl: int | None = 3600,
    ) -> None:
        try:
            import sqlite3

            import sqlite_vec
        except ImportError as exc:
            raise ImportError(
                "Install with `pip install agentic-kit[cache]` to use SqliteVecCache"
            ) from exc
        self._sqlite3 = sqlite3
        self.path = str(path)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                temperature REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
            """
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0("
            f"entry_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        conn.commit()
        self._conn = conn
        self._dim = dim
        return conn

    async def get_by_messages(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
    ) -> CompletionResult | None:
        embedding = await self.embedder.embed(_fingerprint(messages))
        return await self._lookup(embedding, model, temperature)

    async def set_by_messages(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        value: CompletionResult,
        ttl: int | None = None,
    ) -> None:
        embedding = await self.embedder.embed(_fingerprint(messages))
        await self._insert(_fingerprint(messages), embedding, model, temperature, value, ttl)

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
        model: str,
        temperature: float,
    ) -> CompletionResult | None:
        async with self._lock:
            conn = self._connect(len(embedding))
            cur = conn.execute(
                """
                SELECT e.id, e.result_json, e.expires_at, v.distance
                FROM vec_index v JOIN entries e ON e.id = v.entry_id
                WHERE e.model = ? AND e.temperature = ?
                  AND v.embedding MATCH ?
                  AND k = 1
                ORDER BY v.distance
                LIMIT 1
                """,
                (model, round(temperature, 3), _serialize_vector(embedding)),
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
        model: str,
        temperature: float,
        value: CompletionResult,
        ttl: int | None,
    ) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        now = time.time()
        expires_at: float | None = now + effective_ttl if effective_ttl is not None else None
        async with self._lock:
            conn = self._connect(len(embedding))
            cur = conn.execute(
                "INSERT INTO entries (model, temperature, fingerprint, result_json, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model,
                    round(temperature, 3),
                    fingerprint,
                    value.model_dump_json(),
                    now,
                    expires_at,
                ),
            )
            entry_id = cur.lastrowid
            conn.execute(
                "INSERT INTO vec_index (entry_id, embedding) VALUES (?, ?)",
                (entry_id, _serialize_vector(embedding)),
            )
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __len__(self) -> int:
        if self._conn is None:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def __repr__(self) -> str:
        return f"SqliteVecCache(path={self.path!r}, threshold={self.similarity_threshold})"

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "threshold": self.similarity_threshold,
            "dim": self._dim,
            "entries": len(self),
        }
