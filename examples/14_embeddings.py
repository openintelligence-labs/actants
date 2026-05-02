"""Local embeddings + cosine similarity search.

Requires ``ollama serve`` running with ``ollama pull nomic-embed-text``.

Run::

    python examples/14_embeddings.py
"""

from __future__ import annotations

import asyncio

from actants import Embeddings


async def main() -> None:
    emb = Embeddings()  # Ollama default: nomic-embed-text

    docs = [
        "Local-first software keeps your data on your device.",
        "Cloud SaaS centralizes data on remote servers.",
        "Pizza is best with thin crust and basil.",
    ]
    query = "Why use local-first software?"

    doc_vecs = await emb.embed(docs)
    q_vec = await emb.embed_one(query)

    scored = sorted(
        [(Embeddings.cosine(q_vec, v), d) for v, d in zip(doc_vecs.vectors, docs, strict=False)],
        reverse=True,
    )
    for score, text in scored:
        print(f"{score:+.3f}  {text}")


if __name__ == "__main__":
    asyncio.run(main())
