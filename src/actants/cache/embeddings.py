from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx


class Embedder(Protocol):
    """Produce a dense vector for a query string."""

    async def embed(self, text: str) -> list[float]: ...


EmbedFn = Callable[[str], Awaitable[list[float]]]


class OllamaEmbedder:
    """Embedder backed by Ollama's /api/embeddings endpoint. Default local option."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._external = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def embed(self, text: str) -> list[float]:
        r = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        r.raise_for_status()
        data = r.json()
        embedding = data.get("embedding") or []
        if not embedding:
            raise RuntimeError("Ollama returned an empty embedding — is the model pulled?")
        return list(embedding)

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()
