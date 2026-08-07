from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class JsonlAppender:
    """Append-only JSONL writer. Flushes per-record so crashes don't lose writes.

    Each ``write(obj)`` serializes ``obj`` (with ``default=str`` for non-JSON types),
    appends a single line, and flushes the OS buffer. Use as a context manager or
    long-lived object — call ``close()`` when done.
    """

    def __init__(self, path: str | Path, *, ensure_ascii: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._ensure_ascii = ensure_ascii

    def write(self, obj: Any) -> None:
        line = json.dumps(obj, default=str, ensure_ascii=self._ensure_ascii)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlAppender:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[Any]:
    """Yield each JSON object from a .jsonl file. Skips blank lines."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
