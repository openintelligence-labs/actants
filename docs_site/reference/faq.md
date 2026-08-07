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

Subclass `BaseLLMProvider` and implement three methods — `complete`,
`stream_events`, and `health` — then set `name` and the capability flags:

```python
from collections.abc import AsyncIterator

from actants import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    FinishDelta,
    StreamEvent,
    TextDelta,
)


class EchoProvider(BaseLLMProvider):
    name = "echo"
    supports_tool_calls = False
    supports_streaming_tools = False

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> CompletionResult:
        return CompletionResult(content=messages[-1].content, model=model, provider=self.name)

    async def stream_events(
        self, messages, model, temperature=0.7, max_tokens=None, *, tools=None, **kwargs
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text=messages[-1].content)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True
```

`stream_events` is the only streaming method you write. `stream` is derived from
it by the base class — do not override it, or actants will refuse the subclass at
class-definition time, since every streaming path reads `stream_events` and your
override would never run.

If your provider only does non-streaming completions, omit `stream_events`
entirely; it is not abstract, and `complete()` keeps working.

The `OllamaProvider` is a full reference implementation:
<https://github.com/openintelligence-labs/actants/blob/main/src/actants/llm/ollama.py>.

## How do I pass a provider-specific parameter like `seed` or `top_p`?

Any extra keyword argument to `complete`, `stream`, or `stream_events` is
forwarded verbatim to the provider, and folded into the cache key so two seeds
never share one cached answer:

```python
result = await llm.complete("hello", seed=42, top_p=0.9)
```

actants does not validate these names — an unrecognised one surfaces as the
provider's own error.

## Where is the changelog?

[Changelog](changelog.md), or
[`CHANGELOG.md`](https://github.com/openintelligence-labs/actants/blob/main/CHANGELOG.md)
in the repo.

## License

MIT —
[LICENSE](https://github.com/openintelligence-labs/actants/blob/main/LICENSE).
