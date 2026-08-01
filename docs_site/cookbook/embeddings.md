# Cookbook — Semantic search over your notes

Embed a folder of Markdown files locally, search by meaning, no API keys.

```python
import asyncio
from pathlib import Path

from actants import Embeddings, open_sqlite, app_data_dir


def chunk(text: str, *, size: int = 800) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def index(folder: Path) -> None:
    emb = Embeddings()  # nomic-embed-text via Ollama
    db = app_data_dir("notes-search") / "index.db"

    with open_sqlite(db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks(
              id INTEGER PRIMARY KEY,
              path TEXT, body TEXT, vector BLOB
            )
        """)
        for md in folder.rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            for piece in chunk(text):
                v = await emb.embed_one(piece)
                conn.execute(
                    "INSERT INTO chunks(path, body, vector) VALUES (?, ?, ?)",
                    (str(md), piece, _vec_to_bytes(v)),
                )


async def search(query: str, *, k: int = 5) -> None:
    emb = Embeddings()
    q = await emb.embed_one(query)

    db = app_data_dir("notes-search") / "index.db"
    with open_sqlite(db) as conn:
        rows = conn.execute("SELECT id, path, body, vector FROM chunks").fetchall()

    scored = sorted(
        ((Embeddings.cosine(q, _bytes_to_vec(row["vector"])), row) for row in rows),
        key=lambda x: x[0],
        reverse=True,
    )
    for score, row in scored[:k]:
        print(f"{score:.3f}  {row['path']}")
        print(f"  {row['body'][:120]}...")
        print()


def _vec_to_bytes(v: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(v)}f", *v)


def _bytes_to_vec(b: bytes) -> list[float]:
    import struct

    return list(struct.unpack(f"{len(b) // 4}f", b))


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "index":
        asyncio.run(index(Path(sys.argv[2])))
    else:
        asyncio.run(search(" ".join(sys.argv[2:])))
```

```bash
ollama pull nomic-embed-text
python notes_search.py index ~/notes
python notes_search.py search "vector databases vs sqlite-vec"
```

## What this demonstrates

- **Local embeddings** via Ollama — no API key, runs offline
- **SQLite as a vector store** — small enough that you can read every line
- **Linear scan + cosine** — fast enough for tens of thousands of chunks; for
  more, add `sqlite-vec` and an HNSW index

## Where to take it next

- Replace the linear scan with `sqlite-vec` (under the `[cache]` extra) for
  ANN-speed retrieval.
- Wrap as an MCP server: register `search` as a tool and `serve(agent)` so
  Claude Desktop can query your notes.
- Add a re-rank step using a small Ollama model.
