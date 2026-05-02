"""Local-first storage primitives: SQLite + JSONL.

Run::

    python examples/13_local_storage.py
"""

from __future__ import annotations

from actants import JsonlAppender, app_data_dir, open_sqlite, read_jsonl


def main() -> None:
    data_dir = app_data_dir("actants-demo")

    db_path = data_dir / "events.db"
    with open_sqlite(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, kind TEXT, ts TEXT)"
        )
        conn.execute("INSERT INTO events (kind, ts) VALUES (?, ?)", ("startup", "2026-04-26"))
        conn.execute("INSERT INTO events (kind, ts) VALUES (?, ?)", ("shutdown", "2026-04-26"))

    with open_sqlite(db_path) as conn:
        rows = list(conn.execute("SELECT id, kind, ts FROM events"))
        print("SQLite rows:")
        for r in rows:
            print(" ", dict(r))

    log_path = data_dir / "events.jsonl"
    with JsonlAppender(log_path) as log:
        log.write({"event": "tool_called", "name": "search"})
        log.write({"event": "completion", "tokens": 142})

    print("\nJSONL records:")
    for record in read_jsonl(log_path):
        print(" ", record)

    print(f"\nFiles written under: {data_dir}")


if __name__ == "__main__":
    main()
