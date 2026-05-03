from __future__ import annotations

import pytest

from actants.storage import JsonlAppender, open_sqlite, read_jsonl


def test_open_sqlite_creates_with_wal_and_commits(tmp_path):
    db = tmp_path / "test.db"
    with open_sqlite(db) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO t (name) VALUES ('a')")
        conn.execute("INSERT INTO t (name) VALUES ('b')")

    # Reopen, confirm rows persisted
    with open_sqlite(db) as conn:
        rows = list(conn.execute("SELECT name FROM t ORDER BY id"))
        assert [r["name"] for r in rows] == ["a", "b"]


def test_open_sqlite_rolls_back_on_exception(tmp_path):
    db = tmp_path / "rollback.db"
    with open_sqlite(db) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'committed')")

    with pytest.raises(RuntimeError), open_sqlite(db) as conn:
        conn.execute("INSERT INTO t VALUES (2, 'rolled-back')")
        raise RuntimeError("boom")

    with open_sqlite(db) as conn:
        rows = list(conn.execute("SELECT name FROM t"))
        assert [r["name"] for r in rows] == ["committed"]


def test_jsonl_appender_writes_one_line_per_record(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlAppender(path) as app:
        app.write({"a": 1})
        app.write({"b": "two"})
        app.write({"c": [1, 2, 3]})

    records = list(read_jsonl(path))
    assert records == [{"a": 1}, {"b": "two"}, {"c": [1, 2, 3]}]


def test_jsonl_handles_non_json_types_via_default_str(tmp_path):
    from datetime import datetime

    path = tmp_path / "with-dates.jsonl"
    ts = datetime(2026, 4, 26, 12, 0, 0)
    with JsonlAppender(path) as app:
        app.write({"ts": ts})

    records = list(read_jsonl(path))
    assert records[0]["ts"].startswith("2026-04-26")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "blanks.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n\n')
    records = list(read_jsonl(path))
    assert records == [{"a": 1}, {"b": 2}]
