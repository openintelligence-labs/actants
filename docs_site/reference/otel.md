# OpenTelemetry GenAI conformance

actants emits spans following the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), semconv v1.40.0+.

## Span hierarchy

A typical agent run produces:

```
invoke_agent llama3.2          (CLIENT, parent)
├── chat llama3.2              (CLIENT)
├── execute_tool search        (INTERNAL)
├── chat llama3.2              (CLIENT)
└── execute_tool fetch_url     (INTERNAL)
```

## Operation names

`gen_ai.operation.name` always uses one of the spec's enum values:

- `chat` — LLM chat completion
- `embeddings` — embedding request
- `execute_tool` — tool dispatch
- `invoke_agent` — agent run
- `create_agent` — agent construction (we don't currently emit; reserved)

## Attributes we emit

For `chat` spans:

- `gen_ai.operation.name` — `"chat"`
- `gen_ai.provider.name` — e.g. `"ollama"`, `"openai"`, `"anthropic"`
- `gen_ai.request.model` — e.g. `"llama3.2"`
- `gen_ai.request.max_tokens`, `temperature`, `top_p`
- `gen_ai.request.stream` — bool
- `gen_ai.response.model`, `response.id`, `response.finish_reasons`
- `gen_ai.usage.input_tokens`, `output_tokens`, `cache_read.input_tokens`, `cache_creation.input_tokens`
- `gen_ai.response.time_to_first_chunk` — seconds, double
- `gen_ai.conversation.id` — for cross-trace stitching

For `execute_tool` spans:

- `gen_ai.tool.name`, `tool.call.id`, `tool.description`

For `invoke_agent` spans:

- `gen_ai.agent.name`, `agent.id`, `gen_ai.conversation.id`

## Cost — namespaced under `actants.cost.usd`

The OTel GenAI spec does NOT define a cost attribute. actants emits cost under our own namespace so a future `gen_ai.cost.*` won't collide:

```
actants.cost.usd          (float, USD)
```

If the spec later defines a cost attribute, actants will migrate with a deprecation period.

## Stability env-var

The GenAI semantic conventions are still marked **Development**. actants follows current spec by default. To opt into newer experimental attributes:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

If GenAI is promoted to Stable, the default will follow the stable conventions.

## Compatible backends

Anything that consumes OTel spans:

- [Phoenix](https://phoenix.arize.com/) (open source)
- [Langfuse](https://langfuse.com/) (open source)
- [Logfire](https://logfire.pydantic.dev/) (Pydantic's hosted)
- [Datadog](https://docs.datadoghq.com/tracing/), [Honeycomb](https://www.honeycomb.io/), Grafana Tempo, Jaeger
- Any OTLP-compatible collector

## Setting up

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Now any agent.run() will emit spans through this provider.
```

For OTLP export to a real backend, swap `ConsoleSpanExporter` for `OTLPSpanExporter`.
