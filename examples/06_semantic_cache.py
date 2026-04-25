"""Semantic cache backed by sqlite-vec + Ollama embeddings.

Prereqs:
- `ollama pull nomic-embed-text`
- `pip install 'agentic-kit[cache]'`

Run: `python examples/06_semantic_cache.py`
"""

from __future__ import annotations

import asyncio

from agentic_kit import LLM
from agentic_kit.cache.embeddings import OllamaEmbedder
from agentic_kit.cache.semantic import SqliteVecCache


async def main() -> None:
    cache = SqliteVecCache(
        path="/tmp/agentic_kit_demo.db",
        embedder=OllamaEmbedder(model="nomic-embed-text"),
        similarity_threshold=0.15,
    )

    async def make_llm() -> LLM:
        return LLM(cache=cache)

    llm = await make_llm()
    # First call — cache miss, hits Ollama
    r1 = await llm.complete("What is the capital of France?")
    print("First:", r1.content, "latency:", round(r1.latency_ms), "ms")

    # Semantically similar phrasing — should hit the semantic cache
    r2 = await llm.complete("France capital — what is it?")
    print("Second:", r2.content, "latency:", round(r2.latency_ms), "ms")


if __name__ == "__main__":
    asyncio.run(main())
