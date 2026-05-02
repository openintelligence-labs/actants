from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    provider: str
    dimensions: int


class BaseEmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def embed(self, texts: list[str], *, model: str) -> EmbeddingResult: ...

    @abstractmethod
    async def health(self) -> bool: ...
