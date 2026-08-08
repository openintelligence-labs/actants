"""Durable checkpoints for `Agent` runs.

A checkpoint is the whole conversation state of one in-flight run, keyed by a
caller-chosen ``thread_id``. Writing one after every tool result is what lets
`resume` pick a crashed run back up without paying
for the LLM calls again and — more importantly — without re-running side effects that
already happened.

See `resume` for the exact replay guarantee.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from actants.errors import CheckpointSchemaMismatch
from actants.llm.base import ChatMessage, CompletionResult, ToolCall
from actants.storage.sqlite import open_sqlite

if TYPE_CHECKING:
    from actants.agents.agent import AgentStep

#: On-disk schema version, stored in ``PRAGMA user_version``.
#:
#: Bumped whenever the table layout or the JSON shape of a persisted column changes.
#: Unlike the semantic cache — which is disposable and resets itself on a mismatch — a
#: checkpoint store holds the only record of which side effects have already run, so an
#: unreadable file is always an error rather than something to silently discard.
SCHEMA_VERSION = 1

#: Where a run stood when its last checkpoint was written.
#:
#: ``"running"`` is the only status resume will continue from; the other three are
#: terminal in the sense that resume either returns the stored result, re-raises, or —
#: for ``"interrupted"`` — needs an explicit approve/reject decision.
CheckpointStatus = Literal["running", "interrupted", "completed", "failed"]


class StepRecord(BaseModel):
    """One `AgentStep`, in a form that survives a round-trip.

    ``AgentStep`` is a dataclass holding pydantic models; this is the pydantic mirror of
    it, so the whole checkpoint serializes with a single ``model_dump_json``.
    """

    index: int
    completion: CompletionResult
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[str] = Field(default_factory=list)


class Checkpoint(BaseModel):
    """The durable state of one agent run.

    ``messages`` is the full working history — everything the resumed run needs to
    reconstruct its turn, including the tool results that already landed. ``pending_call``
    is set only for ``status="interrupted"``: it is the call the agent stopped in front of
    and has *not* dispatched.
    """

    thread_id: str
    status: CheckpointStatus = "running"
    messages: list[ChatMessage] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    #: Index of the step the run was executing when this checkpoint was written.
    step_index: int = 0
    #: The step budget the original ``run()`` was given, so resume continues under the
    #: same cap rather than silently granting a fresh one.
    max_steps: int = 0
    pending_call: ToolCall | None = None
    tag: str | None = None
    #: Set when ``status="failed"``: the ``repr``-ish description of what went wrong.
    #: Kept as a string because the original exception is not portable across processes.
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


@runtime_checkable
class Checkpointer(Protocol):
    """Where `Agent` persists and reads run state.

    Four async methods, keyed by ``thread_id``. Implement this against any store —
    Redis, Postgres, an object store — and pass it as ``Agent(checkpointer=...)``.

    ``put`` must be a full overwrite of the thread's state, not an append: the agent
    always hands over the complete `Checkpoint`, and resume only ever reads the
    latest one.

    Concurrent access to *different* thread_ids must be safe. Two writers on the *same*
    thread_id is undefined — see the note on
    `resume`.
    """

    async def put(self, checkpoint: Checkpoint) -> None:
        """Store ``checkpoint``, replacing any earlier state for its ``thread_id``."""
        ...

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Return the stored checkpoint for ``thread_id``, or None if there is none."""
        ...

    async def list_threads(self) -> list[str]:
        """Return every ``thread_id`` this store holds."""
        ...

    async def delete(self, thread_id: str) -> bool:
        """Drop ``thread_id``'s state; return whether anything was there to drop."""
        ...


class InMemoryCheckpointer:
    """Process-local checkpointer: durable across a crash of the *run*, not the process.

    Useful for tests and for interrupt/approve flows inside one process. Use
    `SqliteCheckpointer` for anything that must survive the interpreter exiting.
    """

    def __init__(self) -> None:
        self._threads: dict[str, Checkpoint] = {}

    async def put(self, checkpoint: Checkpoint) -> None:
        # Copied on the way in: the agent keeps mutating the object it handed over as the
        # run proceeds, and a stored reference would follow those mutations — so a "crash"
        # after this write would still expose state that was never durably recorded.
        self._threads[checkpoint.thread_id] = checkpoint.model_copy(deep=True)

    async def get(self, thread_id: str) -> Checkpoint | None:
        stored = self._threads.get(thread_id)
        return stored.model_copy(deep=True) if stored is not None else None

    async def list_threads(self) -> list[str]:
        return list(self._threads)

    async def delete(self, thread_id: str) -> bool:
        return self._threads.pop(thread_id, None) is not None

    def __len__(self) -> int:
        return len(self._threads)


class SqliteCheckpointer:
    """Checkpointer backed by a SQLite file, safe to open from several processes.

    Each call opens its own connection through
    `open_sqlite`, so a checkpointer instance carries no
    connection state and separate processes writing *different* thread_ids coexist on
    WAL. Nothing is cached in memory, which is the point: a resume in a fresh process
    reads exactly what the crashed one committed.

    The file records its schema version in ``PRAGMA user_version``; opening one written
    by an incompatible actants raises `CheckpointSchemaMismatch`
    rather than resetting, because these rows are the only record of which side effects
    already ran.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def path(self) -> Path:
        """The database file this checkpointer is bound to. **Read-only.**"""
        return self._path

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with open_sqlite(self._path) as conn:
            self._apply_schema_version(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
        self._initialized = True

    def _apply_schema_version(self, conn: sqlite3.Connection) -> None:
        found = conn.execute("PRAGMA user_version").fetchone()[0]
        has_table = (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
            ).fetchone()[0]
            > 0
        )
        if found == SCHEMA_VERSION:
            return
        if not has_table:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        raise CheckpointSchemaMismatch(
            f"Checkpoint store at {str(self._path)!r} was written by a different actants "
            f"checkpoint schema (found version {found}, this build expects "
            f"{SCHEMA_VERSION}). Its rows record which tool side effects have already "
            "run, so actants will not guess at their layout — resuming from a "
            "misread checkpoint could re-run a side effect that already happened. "
            "Finish or abandon those threads with the actants version that wrote them, "
            "then delete the file."
        )

    async def put(self, checkpoint: Checkpoint) -> None:
        # A single lock for the whole file, not one per thread_id: SQLite serializes
        # writers anyway, and holding it across the blocking sqlite call keeps a
        # concurrently-resumed thread from tripping the busy timeout inside one process.
        async with self._lock:
            await asyncio.to_thread(self._put_sync, checkpoint)

    def _put_sync(self, checkpoint: Checkpoint) -> None:
        self._ensure_schema()
        with open_sqlite(self._path) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                    (thread_id, status, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.thread_id,
                    checkpoint.status,
                    checkpoint.model_dump_json(),
                    checkpoint.created_at,
                    checkpoint.updated_at,
                ),
            )

    async def get(self, thread_id: str) -> Checkpoint | None:
        return await asyncio.to_thread(self._get_sync, thread_id)

    def _get_sync(self, thread_id: str) -> Checkpoint | None:
        self._ensure_schema()
        with open_sqlite(self._path) as conn:
            row = conn.execute(
                "SELECT payload FROM checkpoints WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate(json.loads(row["payload"]))

    async def list_threads(self) -> list[str]:
        return await asyncio.to_thread(self._list_sync)

    def _list_sync(self) -> list[str]:
        self._ensure_schema()
        with open_sqlite(self._path) as conn:
            rows = conn.execute("SELECT thread_id FROM checkpoints ORDER BY updated_at").fetchall()
        return [str(r["thread_id"]) for r in rows]

    async def delete(self, thread_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, thread_id)

    def _delete_sync(self, thread_id: str) -> bool:
        self._ensure_schema()
        with open_sqlite(self._path) as conn:
            cur = conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            return cur.rowcount > 0

    def __repr__(self) -> str:
        return f"SqliteCheckpointer(path={str(self._path)!r})"


def step_to_record(step: AgentStep) -> StepRecord:
    """Convert a live `AgentStep` into its persisted form."""
    return StepRecord(
        index=step.index,
        completion=step.completion,
        tool_calls=list(step.tool_calls),
        tool_results=list(step.tool_results),
    )


def record_to_step(record: StepRecord) -> AgentStep:
    """Rebuild an `AgentStep` from its persisted form."""
    from actants.agents.agent import AgentStep

    return AgentStep(
        index=record.index,
        completion=record.completion,
        tool_calls=list(record.tool_calls),
        tool_results=list(record.tool_results),
    )
