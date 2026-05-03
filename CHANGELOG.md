# Changelog

## [0.5.0] - 2026-04-29

The release that delivers the wedge: **local-first, zero-telemetry, async-only, sub-2ms cold import, native MCP + A2A, OTel-conformant.**

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

- **`actants.bench`** — competitor benchmark harness:
  - `python -m actants.bench` outputs Markdown comparison table
  - Measures bare `import`, first-use import, and `Agent()` instantiation across LangChain, LlamaIndex, Pydantic AI, CrewAI, smolagents, Agno, AutoGen.
  - All numbers reproducible — fresh subprocess per sample.

### Changed — performance

- **`actants/__init__.py` is now PEP 562 lazy** — bare `import actants` is ~1 ms median (down from ~140 ms in 0.4.x). Public API is unchanged; symbols load on first attribute access.
- **CI gate**: `tests/test_cold_import.py` enforces bare-import < 50 ms median. Anyone introducing an eager top-level import fails CI immediately.

### Documentation

- **README rewritten** — local-first hero, runnable code without API keys, real benchmark table, "what we won't build" list.
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
