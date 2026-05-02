from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_sqlite(
    path: str | Path,
    *,
    wal: bool = True,
    foreign_keys: bool = True,
    timeout: float = 30.0,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with WAL + FK + safe defaults; commits on exit.

    WAL mode lets readers and writers coexist without blocking. ``timeout`` controls
    how long writers wait for the file lock before raising. Connection is closed
    on exit; transactions auto-commit on a clean exit, rollback on exception.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=timeout, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if wal:
            cur.execute("PRAGMA journal_mode=WAL")
        if foreign_keys:
            cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
