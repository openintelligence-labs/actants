# Changelog

## [Unreleased]

Four architectural defects found by adversarial review. Each changes an API or a
protocol, so each lands before the 1.0 freeze rather than after.

### Changed (breaking)

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
