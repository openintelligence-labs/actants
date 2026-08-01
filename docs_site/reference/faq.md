# FAQ

## Why does the default provider require running Ollama?

So that `pip install actants` followed by `Agent().run("...")` works
without an API key. If you prefer to start with a cloud provider, set
`ACTANTS_PROVIDER=openai` (or use the constructor argument) and supply
the matching API key.

## Does the framework send any telemetry?

No. `actants` makes no outbound calls during import or on user actions
beyond the LLM/tool/protocol calls the user explicitly initiated. If you
enable OpenTelemetry, spans go to whichever collector you configure.

## Why is the API async-only?

We wanted one execution model rather than maintain parallel sync and
async code paths. Apps that need sync can wrap a call:

```python
import asyncio

asyncio.run(agent.run("..."))
```

## How do I add a new LLM provider?

Subclass `BaseLLMProvider` and implement `complete`, `stream`,
`stream_events`, and `health`; set `name` and `supports_tool_calls`. The
`OllamaProvider` is a small reference implementation:
<https://github.com/openintelligence-labs/actants/blob/main/src/actants/llm/ollama.py>.

## Where is the changelog?

[Changelog](changelog.md), or
[`CHANGELOG.md`](https://github.com/openintelligence-labs/actants/blob/main/CHANGELOG.md)
in the repo.

## License

MIT —
[LICENSE](https://github.com/openintelligence-labs/actants/blob/main/LICENSE).
