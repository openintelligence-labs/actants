# Changelog

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
- `ToolCall`, `ToolSpec` — provider-agnostic tool descriptions in `agentic_kit.llm.base`.
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
