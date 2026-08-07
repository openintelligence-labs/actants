"""Response caching: the request model, the two backend protocols, and two backends.

``SqliteVecCache`` and ``OllamaEmbedder`` are lazy-imported (PEP 562) because they
need the ``[cache]`` extra. They were previously reachable through ``__getattr__`` but
absent from ``__all__``, which meant ``from actants.cache import SqliteVecCache``
failed under ``mypy --strict`` with "does not explicitly export attribute" — the same
defect that `py.typed` was found to have at the top level. The ``TYPE_CHECKING``
re-export block below fixes it the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actants.cache.memory import InMemoryCache, make_key
from actants.cache.protocol import CacheBackend, RequestCacheBackend
from actants.cache.request import KEY_VERSION, CacheRequest

#: Public name → module providing it. Both need `pip install actants[cache]`.
_LAZY: dict[str, str] = {
    "SqliteVecCache": "actants.cache.semantic",
    "CacheSchemaMismatch": "actants.cache.semantic",
    "OllamaEmbedder": "actants.cache.embeddings",
    "Embedder": "actants.cache.embeddings",
}

__all__ = [
    "KEY_VERSION",
    "CacheBackend",
    "CacheRequest",
    "CacheSchemaMismatch",
    "Embedder",
    "InMemoryCache",
    "OllamaEmbedder",
    "RequestCacheBackend",
    "SqliteVecCache",
    "make_key",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is not None:
        from importlib import import_module

        value = getattr(import_module(module_path), name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module 'actants.cache' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    # Type-checker only; resolved at runtime via __getattr__ above.
    from actants.cache.embeddings import Embedder as Embedder
    from actants.cache.embeddings import OllamaEmbedder as OllamaEmbedder
    from actants.cache.semantic import CacheSchemaMismatch as CacheSchemaMismatch
    from actants.cache.semantic import SqliteVecCache as SqliteVecCache
