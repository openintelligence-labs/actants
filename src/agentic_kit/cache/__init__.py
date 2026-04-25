from agentic_kit.cache.memory import InMemoryCache, make_key
from agentic_kit.cache.protocol import CacheBackend

__all__ = ["CacheBackend", "InMemoryCache", "make_key"]


def __getattr__(name: str):
    if name == "SqliteVecCache":
        from agentic_kit.cache.semantic import SqliteVecCache

        return SqliteVecCache
    if name == "OllamaEmbedder":
        from agentic_kit.cache.embeddings import OllamaEmbedder

        return OllamaEmbedder
    raise AttributeError(name)
