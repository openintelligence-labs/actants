# Cache

Every cache backend is keyed by a `CacheRequest`, which carries everything about a
request that can change the answer — provider, model, temperature, `max_tokens`, tool
definitions, and response format, as well as the messages. Exact-match backends
implement `CacheBackend` and receive `CacheRequest.key()`; semantic backends implement
`RequestCacheBackend` and receive the whole request, so they can match message content
by embedding distance while still matching everything else exactly.

::: actants.cache.request.CacheRequest

::: actants.cache.memory.InMemoryCache

::: actants.cache.protocol.CacheBackend

::: actants.cache.protocol.RequestCacheBackend

::: actants.cache.semantic.SqliteVecCache
