# Observability

Two complementary primitives:

- **structlog** for human/JSON logs
- **OpenTelemetry GenAI** spans for distributed traces and cost

## Logging

```python
from actants import setup_logging, get_logger

setup_logging(level="info", format="pretty")  # or "json" for log aggregators
log = get_logger(__name__)
log.info("agent started", model="llama3.2")
```

`setup_logging` is **idempotent** — call it once at process startup. The CLI
helper module (`actants.cli.common_options`) calls it for you when you wire
up `--debug/--quiet/--log-format`.

For MCP stdio servers, **always log to stderr** (the default). stdout is
reserved for the JSON-RPC stream.

## Tracing

actants emits OpenTelemetry GenAI semantic-conventions-conformant spans
(semconv v1.40.0+):

```
invoke_agent llama3.2          (CLIENT, parent)
├── chat llama3.2              (CLIENT)
├── execute_tool search        (INTERNAL)
├── chat llama3.2              (CLIENT)
└── execute_tool fetch_url     (INTERNAL)
```

All `gen_ai.*` attributes match the spec exactly. Cost (which the spec does
not define) is namespaced under `actants.cost.usd`.

See [OTel GenAI](../reference/otel.md) for the full attribute catalog and
backend setup. Compatible with **Phoenix**, **Langfuse**, **Logfire**,
**Datadog**, **Honeycomb**, Jaeger, Tempo, or any OTLP collector.

## Cost tracking

```python
from actants import LLM, CostTracker

tracker = CostTracker()
llm = LLM(cost_tracker=tracker)
await llm.complete("hello")
print(tracker.total_usd, tracker.records[-1])
```

Costs are computed from the versioned pricing table at
`actants.cost.pricing` and attached to OTel spans automatically.
