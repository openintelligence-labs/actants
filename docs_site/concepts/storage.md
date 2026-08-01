# Storage

Two append-only primitives — SQLite and JSONL — plus XDG-aware paths. That's
the storage layer; everything else is an app pattern you write yourself.

## SQLite with safe defaults

```python
from actants import open_sqlite

with open_sqlite("notes.db") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO notes(body) VALUES (?)", ("first note",))
```

`open_sqlite` configures:

- **WAL** journal mode (concurrent readers + writers)
- **Foreign keys** enabled
- **Row factory** = `sqlite3.Row` (dict-like access)
- **Synchronous=NORMAL** (durable + fast)
- **Auto-commit** on clean exit, **rollback** on exception
- Parent directory created if missing

For semantic search, install `[cache]` and use `sqlite-vec` directly inside the
connection — actants does not wrap it because the API is small and the
schema is your decision.

## Append-only JSONL

```python
from actants import JsonlAppender, read_jsonl

with JsonlAppender("events.jsonl") as out:
    out.write({"event": "agent_started", "model": "llama3.2"})
    out.write({"event": "tool_called", "name": "search"})

for record in read_jsonl("events.jsonl"):
    print(record)
```

Each `write()` flushes immediately, so a crash mid-loop loses at most the
current record (not all buffered ones).

## Per-user paths

```python
from actants import app_data_dir

db_path = app_data_dir("deepdive") / "notes.db"
```

See [Configuration → Per-user paths](../configuration.md#per-user-paths-xdg-aware).

## What we don't ship

- Vector DB integrations beyond SQLite — `sqlite-vec` covers local-scale search; if you
  outgrow it, wire up your own vector store
- ORM helpers — write SQL or use SQLAlchemy on top
- Migration framework — use [Alembic](https://alembic.sqlalchemy.org/) or your own
  migration scripts
