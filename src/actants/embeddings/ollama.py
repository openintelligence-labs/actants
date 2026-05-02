from __future__ import annotations

import httpx

from actants.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def embed(self, texts: list[str], *, model: str = "nomic-embed-text") -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=model, provider=self.name, dimensions=0)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        vectors = data.get("embeddings") or []
        dim = len(vectors[0]) if vectors else 0
        return EmbeddingResult(
            vectors=vectors,
            model=model,
            provider=self.name,
            dimensions=dim,
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
