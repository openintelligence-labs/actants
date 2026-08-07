# Changelog

## [Unreleased]

## [1.0.0] - 2026-08-06

**The API is now stable.** See the
[stability policy](https://actants.openintelligence-labs.org/reference/stability/) for
what that covers: `actants.__all__` is the public API, 1.x will not break it, and
anything provisional (`mcp`, `a2a`, `bench`, the cache file format, OTel attribute
names) says so explicitly.

The honest version of how this release came about: 1.0 was not reached by adding
features. It was reached by an adversarial review of the existing code that found a
row of defects sharing one property — **they were all silent**. Tools passed to a
provider that could not call them were dropped, and the model answered as though no
tools existed. The semantic cache collided across requests that differed in
`max_tokens`, so a 4096-token answer could be served to a request capped at 16. Every
streamed run reported a cost of $0.00. A fallback mid-stream spliced two completions
into one response. `seed=42` type-checked, looked like it worked, and set no seed.
None of these raised; each one just returned a plausible wrong answer. Fixing them is
the substance of this release, and it is why the version number moved.

The API hardening that follows was the second half: a 1.0 promise is only worth making
about a surface that is worth freezing, so anything that would have been painful to
live with for the life of 1.x — an unnormalized provider string in a
provider-agnostic type, a cost tag that vanished on the streaming path, mutable
attributes that lied after assignment — was fixed *before* the freeze rather than
deprecated after it.

Also in 1.0: **13 providers** (Ollama, OpenAI, Anthropic, Gemini, plus nine
OpenAI-compatible hosts behind one generated adapter), **corrected pricing with honest
unknown-cost reporting** — an unpriced model now surfaces in
`CostTracker.untracked_models` and flags the total as a lower bound, instead of
contributing $0.00 to a total that looked authoritative — a **published benchmark**
with full methodology and reproduction commands, and documentation whose every code
block is executed as a test.

### Added

- **`tag` on every path that spends money.** `LLM.stream`, `LLM.stream_events`,
  `LLM.extract`, `LLM.extract_stream`, and `LLM.run_agent_stream` now accept `tag`,
  matching `LLM.complete` and `Agent.run` / `Agent.stream`. Cost attribution
  previously vanished the moment a user switched a tagged `complete()` call to
  streaming: the spend still reached `total_usd`, so nothing looked wrong, but
  `by_tag` quietly stopped adding up.

  Streamed spend is recorded once per request, when the provider reports its
  `UsageDelta`, through the same `CostTracker.record()` that `complete()` uses — so an
  unpriced model streamed under a tag also lands in `untracked_models`, exactly as a
  completion does. A stream abandoned before the usage event records nothing, because
  actants never saw what it cost. Recording now happens in one place for all streaming
  entry points; `Agent.stream` no longer records separately, which also removes a
  latent double-count.

- **`CompletionResult.raw_finish_reason`** preserves the provider's own stop-reason
  string verbatim, alongside the normalized `finish_reason`. Nothing is lost by
  normalization.

- **`FinishReason`, `FINISH_REASONS`, and `normalize_finish_reason`** are exported at
  top level, since `FinishReason` is now the type of a public field.

### Changed (breaking)

- **`finish_reason` is normalized across providers.** It was `str | None` carrying
  whatever the provider said — `"stop"` from OpenAI, `"end_turn"` from Anthropic,
  `"STOP"` from Gemini, `"done"`-style values from Ollama — inside a deliberately
  provider-agnostic result type, so callers could not branch on it portably without
  writing the union of every provider's vocabulary. It is now
  `Literal["stop", "length", "tool_calls", "content_filter", "error", "unknown"]`,
  mapped explicitly for every provider, with the raw string preserved on
  `raw_finish_reason`. `FinishDelta` gains the same treatment (`reason` normalized,
  `raw_reason` verbatim) so streamed and completed runs can be branched on identically.

  This had to land before 1.0 rather than after: a field can be *widened* in a minor
  release, but never narrowed, so `str | None` would have been permanent.

  An unrecognized provider value maps to `"unknown"` and keeps its raw string — it is
  never an exception. Providers extend these enums without notice (Gemini alone has
  added `MALFORMED_FUNCTION_CALL`, `BLOCKLIST`, `SPII`, and `IMAGE_SAFETY` since
  launch), and a completion that already succeeded must not be turned into a crash by
  a string actants has not seen before. `FinishReason` is an **open** set: new
  canonical values may be added in a minor release, so give it a default branch.

  Migration: `result.finish_reason == "stop"` keeps working for OpenAI-family
  providers and now *also* works for Anthropic, Gemini, and Ollama. Code matching a
  provider-native spelling (`== "end_turn"`, `== "MAX_TOKENS"`) should either switch to
  the canonical value or read `raw_finish_reason`. The field is no longer `None` when
  absent — it is `"unknown"`.

- **`SqliteVecCache.path` and `.embedder` are read-only properties.** Both were plain
  attributes, but both are snapshotted into the sqlite connection on first use, so
  assigning them moved nothing while `describe()` and `repr()` went on reporting the
  new value — the cache would claim to be a file it was not using. Swapping the
  embedder was worse than a no-op: vectors from two embedders are not comparable, so
  it would have compared a new model's vectors against an old model's index. Assigning
  either now raises `AttributeError`. Construct a new cache instead.

### Documentation

- **A stability policy** (`docs_site/reference/stability.md`, linked from the README)
  states what the 1.0 guarantee covers: the public API is exactly `actants.__all__`,
  what each semver level permits, which surfaces are provisional and why, what is
  explicitly *not* promised (model output, costs for a given prompt, cache hit rates,
  exception message wording), and the deprecation process — `DeprecationWarning` plus
  at least two minor releases and six months before any removal, which only ever
  happens in a major version.

- **`max_repairs` counting is documented against `max_attempts` and `max_steps`.** The
  three were compared and the difference is deliberate, not an off-by-one:
  `max_attempts` and `max_steps` bound the **total**, while `max_repairs` bounds the
  **extras** — the initial completion always happens, plus at most `N` self-corrections
  after it. A repair is not a retry; it sends a new, longer conversation containing the
  model's bad output and the parser error. Naming it `max_attempts` would have implied
  `1` allows one self-correction when it would in fact allow none. The `LLM` class
  docstring now states all three conventions in one place, and tests pin the counts.

---

The remainder of this entry covers the correctness and API work done in the run-up to
the freeze, all of it released here for the first time.

### Changed (breaking)

- **`stream_events` is the single streaming primitive a provider implements.**
  `LLM.stream()` filters `stream_events` rather than calling `provider.stream()`, so a
  third-party provider that overrode only `stream()` was silently never called — it
  appeared to stream nothing. `stream` is now a concrete helper the base class derives
  from `stream_events`, and `BaseLLMProvider.__init_subclass__` rejects a subclass that
  overrides `stream` without `stream_events`, naming the rename. `stream_events` is
  deliberately not abstract, so a completion-only provider needs no streaming stub;
  the default raises `NotImplementedError` only if something tries to stream.

  Custom providers overriding `stream()` must rename it to `stream_events()` and yield
  `TextDelta` / `FinishDelta` instead of plain strings. Every built-in provider's
  `stream()` was identical boilerplate and has been removed in favour of the helper.

- **The exception hierarchy moved to `actants.errors` and is exported at top level.**
  `from actants import ActantsError, ModelNotFoundError, ToolCallsNotSupportedError`
  now works; the classes were previously reachable only via `actants.llm.errors`.
  `ToolError`, `AllProvidersFailedError`, `CacheSchemaMismatch`, and
  `MCPConnectionError` now inherit `ActantsError` too, so `except ActantsError` is an
  exhaustive catch for actants' own failures — it previously missed all four. Every
  class keeps its builtin base (`UnknownProviderError` is still a `ValueError`,
  `ToolCallsNotSupportedError` still a `TypeError`), and `actants.llm.errors`
  re-exports the identical class objects, so existing imports and `except` clauses are
  unaffected.

### Added

- **Provider-specific parameters pass through and reach the cache key.** `complete`,
  `stream`, and `stream_events` accept arbitrary keyword arguments — `seed`, `top_p`,
  `stop` — forward them verbatim to the provider, and fold them into
  `CacheRequest.extra` so a `seed=1` answer is never served to a `seed=2` request.
  `extra` was documented and hashed but populated by nobody, which armed the collision
  for whenever passthrough was added.

  The providers needed fixing too: all of them already accepted `**kwargs` and
  discarded it, so `provider.complete(..., seed=42)` type-checked, looked like it
  worked, and never set a seed. Ollama and Gemini now route passthrough into `options`
  / `generationConfig` with their genuinely top-level fields handled separately;
  OpenAI and Anthropic forward it flat.

- `Agent(concurrency=...)` and `SqliteVecCache(on_schema_mismatch=...)` are typed
  `Literal` (`ConcurrencyMode`, `SchemaMismatchAction`) rather than bare `str`, so a
  typo is caught by the type checker at the call site instead of at construction.
  Matches how the rest of the API spells a closed string set (`Role`, `LogFormat`,
  `LogLevel`). The runtime check remains for callers without a type checker, and both
  messages now name the valid values.

- `docs_site/api/errors.md` documents the exception hierarchy and how to handle it —
  the errors had no documentation at all.

### Fixed

- **`FallbackProvider` capability flags are derived from the chain on every access.**
  `supports_tool_calls` / `supports_streaming_tools` were computed once in `__init__`,
  so a provider that set its flag afterwards was invisible: a stale `False` refused
  tools the chain could serve, and a stale `True` handed tools to a provider that
  dropped them. Both are now read-only properties; assigning to them raises and points
  at the member provider to set instead. `FallbackProvider.stream` is gone — the base
  class derives it from `stream_events`, which already implements the same fail-over.

- **Cache backends are now keyed by a `CacheRequest`.** The semantic-cache protocol
  (`get_by_messages` / `set_by_messages`) only ever received messages, model, and
  temperature, so `SqliteVecCache` collided across requests that differed in
  `max_tokens`, provider, tool definitions, or response format — a 4096-token answer
  could be served to a request capped at 16. Both cache kinds now take one
  `CacheRequest` carrying every field that changes the answer. Exact-match backends
  implement `CacheBackend` and receive `CacheRequest.key()`; semantic backends implement
  the new `RequestCacheBackend` and receive the whole request, matching message content
  by embedding distance and everything else exactly (`CacheRequest.scope_hash()`).
  `make_key()` keeps its signature and now delegates to `CacheRequest`.

  Custom cache backends implementing `get_by_messages` / `set_by_messages` must rename
  them to `get_request` / `set_request` and read the fields off the request.

- **`SqliteVecCache` on-disk schema is versioned.** The database records its schema in
  `PRAGMA user_version`, tied to the cache key version. Opening a file written by an
  incompatible version discards it and starts empty rather than serving entries keyed on
  fewer fields; pass `on_schema_mismatch="error"` to raise `CacheSchemaMismatch` instead.

- **`Agent` defines its concurrency semantics.** Concurrent `run()` calls previously
  appended to one shared `ConversationMemory` and produced merged history. Each run now
  works against a private copy seeded from the agent's memory and commits its turn back
  atomically, so concurrent runs never observe each other's partial state, and a run that
  raises commits nothing. `Agent(concurrency="serialized")` selects the other contract:
  runs queue on a lock and each sees every turn committed before it.

### Fixed

- `LLM` now checks `supports_tool_calls` / `supports_streaming_tools` before sending
  tools to a provider. Tools passed to a provider that cannot use them were silently
  dropped, so the model answered as if no tools existed. Raises
  `ToolCallsNotSupportedError` naming the provider, the tools, and the fix.
- `Agent.stream()` called `provider.stream_events` directly, bypassing retry, tracing,
  and per-call model/temperature overrides — a streamed run behaved differently from a
  non-streamed one. All streaming (`LLM.stream`, `LLM.stream_events`,
  `LLM.extract_stream`, `LLM.run_agent_stream`, `Agent.stream`) now goes through one
  layered path. Stream retry applies only before the first event reaches the consumer,
  since restarting mid-stream would splice two completions together.
- `SqliteVecCache` scoped its lookup as a post-filter on a KNN query. Because
  `MATCH ... AND k = 1` selects the nearest vectors *before* the rest of the `WHERE`
  clause applies, a nearer entry in another scope displaced the correct one and the
  lookup returned a spurious miss. Scope is now a `vec0` partition key, pruned before
  the search.
- `import actants.cache.semantic` (or `.protocol`) as the first actants import raised
  `ImportError` from a circular import via `actants.llm.__init__`. The protocol module's
  imports are now annotation-only.

### Documentation

- `docs_site/` is now committed rather than gitignored, so the published documentation
  lives in the repository and is covered by CI. The docs-snippet suite checks every
  Python block in `docs_site/` as well as `README.md`, plus a guard that every page is
  reachable from the mkdocs nav.

## [0.5.3] - 2026-07-29

### Fixed

- Pinned `mcp>=1.0,<2` in the `mcp`, `all`, and `dev` extras. The MCP Python
  SDK 2.0.0 removed `mcp.server.fastmcp` (used by `actants.mcp.serve` /
  `build_server`) and `mcp.shared.memory.create_connected_server_and_client_session`,
  breaking `actants.mcp` at first use. The pin stays until `actants.mcp` is
  ported to the 2.x API.
- `LLM(provider="openai")` used to silently store the string as the provider
  and crash later at call time (`'str' object has no attribute 'name'`).
  Provider name strings (`"ollama"`, `"openai"`, ...) are now coerced to the
  matching provider at construction; any other non-`BaseLLMProvider` value
  raises a clear `TypeError` immediately.

### Notes

- This release is not on PyPI yet (publishing is blocked on credentials);
  PyPI still carries 0.5.2. Install 0.5.3 from the GitHub release artifacts.

## [0.5.2] - 2026-05-03

### Documentation

- Rewrote README and several `docs_site/` pages (`index`, `differentiation`,
  `faq`, migration guides) in a neutral technical tone. Removed manifesto
  framing, "what we won't build" lists, and competitor-comparison sections
  that were inappropriate for a pre-1.0 library.

### Note on 0.5.0 and 0.5.1

Both 0.5.0 and 0.5.1 are yanked. They shipped the same code as 0.5.2 but
with documentation that included unverified comparison numbers (0.5.0) and
manifesto-style framing (0.5.1). Use 0.5.2 instead.

## [0.5.1] - 2026-05-03 (yanked)

### Documentation

- Removed comparison-benchmark table and specific timing claims from README
  and docs that lacked verifiable measurement methodology.
- Removed unverified competitor-issue references and quantitative
  "Nx faster" claims from migration guides.

(Yanked: README still contained inappropriate manifesto/positioning sections.
See 0.5.2.)

## [0.5.0] - 2026-04-29 (yanked)

First release of the renamed framework (previously `agentic-kit`). Adds native MCP and A2A integration, an `Agent` class with streaming, OpenTelemetry GenAI conformance, and several app-level helpers (config, CLI, embeddings, storage, testing).

### Added — interop

- **`actants.mcp`** — native Model Context Protocol integration in both directions:
  - `MCPClient({...})` — connect to N MCP servers (stdio + Streamable HTTP), expose their tools to your agent. Config shape mirrors Claude Desktop's `mcpServers`.
  - `MCPClient.tools()` returns a flat list of actants `Tool` objects, name-prefixed by server (`git__status`).
  - `serve(agent)` / `build_server(agent)` — expose your agent's tools as an MCP server in two lines.
  - Uses the official `mcp` Python SDK; under `[mcp]` extra.
- **`actants.a2a`** — native Agent2Agent protocol (Linux Foundation v1.0):
  - `serve(agent, port=9000)` — expose your agent over A2A. Auto-mounts `/.well-known/agent-card.json` + JSON-RPC. AgentCard auto-generated from your tool registry (one skill per tool). Streaming via SSE.
  - `RemoteAgent(url)` — call a remote A2A agent as a local tool. Lazy card resolution.
  - Uses the official `a2a-sdk` Python SDK; under `[a2a]` extra.

### Added — observability

- **`actants.tracing.genai`** — OpenTelemetry GenAI semantic-conventions-conformant spans (semconv v1.40.0+):
  - `chat_span(...)`, `execute_tool_span(...)`, `invoke_agent_span(...)`, `embeddings_span(...)`, `record_response(...)`.
  - All `gen_ai.*` attribute names match the spec exactly (`gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, etc.).
  - Cost is namespaced under `actants.cost.usd` (the spec doesn't define cost).
  - `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` opt-in for newer experimental attributes.
  - Compatible with Phoenix, Langfuse, Logfire, Datadog, any OTel backend.

### Added — agents

- **`Agent.stream(prompt)`** — streaming agent loop yielding typed events:
  - `AgentTextDelta(text, step)` — token-level text
  - `AgentToolCallStarted(call, step)` / `AgentToolCallCompleted(call, value, ok, step)` — tool dispatch lifecycle
  - `AgentStepCompleted(step, completion)` — end of one LLM call + tool round
  - `AgentRunCompleted(content, final)` — final answer (terminal)
  - Memory updates incrementally; final state matches `Agent.run()`.

### Added — tooling

- **`actants.bench`** — internal benchmark harness:
  - `python -m actants.bench` outputs a Markdown comparison table for local measurement.
  - Measures bare `import`, first-use import, and `Agent()` instantiation.
  - Each measurement runs in a fresh subprocess.

### Changed — performance

- **`actants/__init__.py` is now PEP 562 lazy** — module symbols load on first attribute access rather than at import time. Public API is unchanged.
- **CI gate**: `tests/test_cold_import.py` enforces an upper bound on bare-import time so eager top-level imports fail CI.

### Documentation

- **README rewritten** — local-first quickstart and runnable examples without API keys.
- **mkdocs site** under `docs_site/`: quickstart, concepts (Agent / LLM / Tools / Streaming / ReAct), MCP client+server, A2A client+server, cookbook (research-agent, mcp-tools, a2a-pair), migration guides (LangChain, CrewAI, Pydantic AI), reference (differentiation, benchmarks, OTel).

### Known follow-ups (deferred to 0.6+)

- OAuth 2.1 authorization server scaffolding for MCP (clients can connect to OAuth servers; we don't host one yet).
- A2A push notification webhook emission (clients support; servers don't emit).
- Signed Agent Card key management UX (we generate and verify; key rotation is manual).
- MCP resources/prompts/sampling primitives (only tools today).
- Unified `actants.serve(agent, mcp=True, a2a=True)` one-liner.

## [0.4.0] - unreleased

### Added — framework expansion

actants graduates from "LLM gateway" to "local-first AI app framework". Pure additions; no breaking changes from 0.3.

- **`Agent`** — stateful tool-calling agent with `ConversationMemory` and `AgentHooks` (before_step, after_step, on_tool_call, on_error). Wraps `LLM.run_agent` with state and lifecycle.
- **`AppSettings`** — base class for app-level pydantic settings with `.env` loading and per-app env-prefix.
- **`app_config_dir`, `app_data_dir`, `app_cache_dir`** — XDG-aware per-user paths (macOS / Linux / Windows).
- **`setup_logging`, `get_logger`** — one-call structlog configurator (pretty or JSON). OTel tracing remains via the existing `tracing/` module.
- **`make_app`, `common_options`, `console`, `success`, `error`** — Click + Rich helpers under the new `[cli]` extra.
- **`open_sqlite`** — context manager: SQLite with WAL, foreign keys, safe defaults.
- **`JsonlAppender`, `read_jsonl`** — append-only JSONL primitives.
- **`Embeddings`, `OllamaEmbeddingProvider`** — local-first embedding client with cosine helper. Default: `nomic-embed-text` via Ollama.
- **`actants.testing`** — `FakeLLMProvider`, `FakeEmbeddingProvider`, `fake_completion`, `fake_tool_call_completion` for app tests with no network.

### Added — packaging
- `[cli]` extra (`click`, `rich`).
- `[all]` extra now includes `[cli]` deps.

## [0.3.0] - unreleased

### Added
- **Streaming with tool calls.** `provider.stream_events()` + `LLM.stream_events()` yield typed `TextDelta` / `ToolCallDelta` / `UsageDelta` / `FinishDelta` events. `LLM.run_agent_stream()` streams a full agent loop.
- **Partial-JSON streaming.** `LLM.extract_stream(prompt, PydanticModel)` yields progressively-complete pydantic instances as JSON arrives. Forgiving parser tolerates truncated output.
- **New providers:** `GeminiProvider` (native httpx, no SDK), `GroqProvider`, `MistralProvider` — both subclass `OpenAIProvider` via OpenAI-compatible endpoints.
- Pricing added for Gemini 2.5 Pro/Flash, Gemini 1.5, Groq llama-3.3/3.1/Mixtral/qwen-qwq, Mistral Large/Small/Codestral.
- `OpenAIProvider` now accepts `base_url` for self-hosted and compat endpoints.
- `py.typed` marker for full type-checker support.
- `[all]` extra installs every optional dependency.

### Changed
- `BaseLLMProvider` gained `stream_events()` (default wraps `stream()`) and `supports_streaming_tools` flag.

## [0.2.0]

### Added
- `LLM.run_agent()` — tool-calling agent loop across Ollama / OpenAI / Anthropic.
- `LLM.extract()` — structured output into any pydantic model with self-repair.
- `SqliteVecCache` — semantic cache backed by sqlite-vec (under `[cache]` extra).
- `OllamaEmbedder` — local embeddings via Ollama's `/api/embeddings`.
- `RetryPolicy` + `retry_async` — exponential backoff with jitter.
- `FallbackProvider` — chain providers for local-first-with-cloud-fallback resilience.
- `ToolCall`, `ToolSpec` — provider-agnostic tool descriptions in `actants.llm.base`.
- `ToolRegistry.as_specs()` — convert registered tools to provider-agnostic specs.
- Cache, cost tracker, tracing, and retry are now wired into `LLM.complete()` as optional layers.
- Pricing refreshed for Claude 4.7 / Haiku 4.5 / GPT-4.1 / o3.
- Six runnable `examples/` covering the above.

### Changed
- `ChatMessage` gained `tool_call_id` and `tool_calls` fields.
- `CompletionResult` gained a `tool_calls` field.
- `BaseLLMProvider.complete` gained a `tools` parameter; providers declare `supports_tool_calls`.

## [0.1.0]

### Added
- Initial release
- `OllamaProvider` with chat, streaming, and health check
- `OpenAIProvider` (optional `[openai]` extra)
- `AnthropicProvider` (optional `[anthropic]` extra)
- `LLM` high-level client with env-configured defaults
- `CostTracker` + pricing table for OpenAI/Anthropic models
- `InMemoryCache` with TTL
- `ToolRegistry` with async permission checks
- OpenTelemetry span wrapping via `instrument_llm` decorator
