from actants.cache.memory import InMemoryCache, make_key
from actants.cache.protocol import CacheBackend

__all__ = ["CacheBackend", "InMemoryCache", "make_key"]


def __getattr__(name: str):
    if name == "SqliteVecCache":
        from actants.cache.semantic import SqliteVecCache

        return SqliteVecCache
    if name == "OllamaEmbedder":
        from actants.cache.embeddings import OllamaEmbedder

        return OllamaEmbedder
    raise AttributeError(name)
