from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from actants.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from actants.embeddings.ollama import OllamaEmbeddingProvider


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACTANTS_EMBED_", extra="ignore")

    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


def _make_provider(settings: EmbeddingSettings) -> BaseEmbeddingProvider:
    p = settings.provider.lower()
    if p == "ollama":
        return OllamaEmbeddingProvider(base_url=settings.base_url)
    raise ValueError(f"Unknown embedding provider: {settings.provider}")


class Embeddings:
    """High-level embedding client.

    Defaults to Ollama with ``nomic-embed-text`` (768-dim, fast, runs locally).
    Cosine similarity helper provided for convenience::

        emb = Embeddings()
        result = await emb.embed(["hello", "world"])
        score = Embeddings.cosine(result.vectors[0], result.vectors[1])

    ``settings`` is copied at construction, so mutating the object you passed in has no
    effect on this client. Of the copy's own fields, only ``model`` is re-read per call;
    ``provider`` and ``base_url`` select the provider once, at construction. To change
    those, build a new ``Embeddings``.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider | None = None,
        *,
        model: str | None = None,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        # Copied, not aliased — see LLM.__init__ for why.
        self.settings = (
            settings.model_copy(deep=True) if settings is not None else EmbeddingSettings()
        )
        if model is not None:
            self.settings.model = model
        self.provider = provider or _make_provider(self.settings)

    async def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        return await self.provider.embed(texts, model=model or self.settings.model)

    async def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        result = await self.embed([text], model=model)
        return result.vectors[0] if result.vectors else []

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
