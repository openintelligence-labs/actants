from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Annotation-only. Importing ``actants.llm.base`` eagerly pulls in
    # ``actants.llm.__init__`` -> ``actants.llm.client``, which imports this module
    # back — so `import actants.cache.semantic` as the first actants import failed
    # with a partially-initialized-module ImportError.
    from actants.cache.request import CacheRequest
    from actants.llm.base import CompletionResult


class CacheBackend(Protocol):
    """Key-value cache keyed by an opaque string.

    Implement this for exact-match caches. The key is produced by
    `CacheRequest.key`, so it already covers every request parameter that changes
    the answer — a backend must never derive its own key from a subset of the request.
    """

    async def get(self, key: str) -> CompletionResult | None: ...
    async def set(self, key: str, value: CompletionResult, ttl: int | None = None) -> None: ...
    async def clear(self) -> None: ...


@runtime_checkable
class RequestCacheBackend(Protocol):
    """Cache that decides hits from the whole request rather than a precomputed key.

    Semantic caches need this: they match message content by embedding distance, so they
    cannot be handed a hash. They receive the full `CacheRequest` and are
    responsible for honouring *every* field on it — a backend that ignores ``max_tokens``
    or ``tools`` returns answers generated under different constraints.

    `LLM` prefers this protocol when a backend implements it,
    and falls back to `CacheBackend`.
    """

    async def get_request(self, request: CacheRequest) -> CompletionResult | None: ...
    async def set_request(
        self, request: CacheRequest, value: CompletionResult, ttl: int | None = None
    ) -> None: ...
    async def clear(self) -> None: ...
