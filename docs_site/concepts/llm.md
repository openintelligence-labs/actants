# LLM and providers

`LLM` is the provider-agnostic gateway. Same API across Ollama, OpenAI, Anthropic,
Gemini, and every major OpenAI-compatible host — Groq, Mistral, xAI, DeepSeek,
Together, Fireworks, OpenRouter, Cerebras, and Perplexity.

!!! note "Not all 13 providers are live-verified"

    Supporting 13 providers is a claim about code paths, not about how many have been
    run against a live endpoint. Ollama and the shared OpenAI-compatible request path
    are live-verified; the rest are covered by mock-based unit tests whose wire formats
    come from provider documentation. See
    [Provider verification status](https://github.com/openintelligence-labs/actants/blob/main/docs/PROVIDER_VERIFICATION.md)
    for what was measured, and `python -m verification.run` to reproduce it.

## Default: Ollama, no config

```python
from actants import LLM

llm = LLM()  # Ollama at localhost:11434
result = await llm.complete("hello")
print(result.content, result.usage.total_tokens)
```

## Switch providers

```python
LLM()  # Ollama (local)
LLM(provider="openai", model="gpt-4o")  # needs OPENAI_API_KEY
LLM(provider="anthropic", model="claude-3-5-sonnet")  # needs ANTHROPIC_API_KEY
LLM(provider="gemini", model="gemini-2.0-flash")  # needs GOOGLE_API_KEY
LLM(provider="groq", model="llama-3.3-70b-versatile")  # needs GROQ_API_KEY
LLM(provider="mistral", model="mistral-large-latest")  # needs MISTRAL_API_KEY
LLM(provider="xai", model="grok-4")  # needs XAI_API_KEY
LLM(provider="deepseek", model="deepseek-chat")  # needs DEEPSEEK_API_KEY
LLM(provider="together", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")  # TOGETHER_API_KEY
LLM(provider="fireworks", model="accounts/fireworks/models/llama-v3p3-70b-instruct")
LLM(provider="openrouter", model="anthropic/claude-opus-4-8")  # needs OPENROUTER_API_KEY
LLM(provider="cerebras", model="llama-3.3-70b")  # needs CEREBRAS_API_KEY
LLM(provider="perplexity", model="sonar-pro")  # needs PERPLEXITY_API_KEY
```

Or set `ACTANTS_PROVIDER`, `ACTANTS_MODEL`, `ACTANTS_API_KEY` and call `LLM()`.

## Add caching, cost tracking, retry, fallback

```python
from actants import (
    LLM,
    InMemoryCache,
    CostTracker,
    RetryPolicy,
    FallbackProvider,
    OllamaProvider,
)
from actants.llm.openai_provider import OpenAIProvider

llm = LLM(
    provider=FallbackProvider([OllamaProvider(), OpenAIProvider()]),
    cache=InMemoryCache(),
    cost_tracker=CostTracker(),
    retry_policy=RetryPolicy(max_attempts=3, initial_delay=1.0),
)
```

The same primitives compose. Each layer is opt-in.

## Streaming

```python
async for chunk in llm.stream("write a haiku"):
    print(chunk, end="", flush=True)
```

For typed events (text deltas, tool calls, usage, finish), use `llm.stream_events(...)`.

## Structured output

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


person = await llm.extract("John is 30 years old.", Person)
print(person.name, person.age)
```

There are two ways this can be done, and `extract()` picks between them per call.

### The native path

Where the provider can constrain decoding, the JSON Schema goes **on the wire** and
schema-invalid output is not merely unlikely, it is impossible. Every provider spells that
differently — OpenAI takes a `response_format` block, Anthropic has no such parameter and
instead needs a forced single tool call, Gemini nests the schema under
`generationConfig`, Ollama takes it as `format` — so a provider declares its
`native_schema_mode` and actants translates one pydantic model into whatever that mode
needs.

On this path the schema is *not* also repeated in the system prompt: it would spend tokens
on every call restating something the decoder is already enforcing. The repair loop is
never entered either, because a schema-valid response cannot fail to parse.

Strict mode (OpenAI and Anthropic) needs one translation to keep that last sentence true.
It has no way to say "this field may be absent" — every property must be listed in
`required` — so the encoding for optional is a union with `null`. A field with a non-null
default, `priority: int = 3`, therefore goes on the wire as `["integer", "null"]` **and**
required, and a provider doing exactly what it was told may answer `null`. That is not a
value the pydantic model accepts; it is the absence the widening was standing in for. So
`extract` reads a `null` for one of those fields back as absence and lets the field's
default apply. A field you genuinely declared `str | None` keeps its `null` as a real
value — only fields this rewrite *made* nullable are treated this way.

### The prompt path

When the provider has no native mode, the schema is described in a system prompt and a
response that fails to parse is fed back for repair. `max_repairs` counts repair attempts,
not total attempts: `max_repairs=1` (the default) allows one self-correction after the
initial completion, for at most two requests. `max_repairs=0` disables repair.

The prompt path is not a legacy fallback that is being phased out. It is also what runs
when a schema turns out to be inexpressible in the provider's dialect — a recursive model,
or a bare `dict` field with no declared properties, which strict mode cannot represent
because it forbids the open-ended `additionalProperties` that would describe it. `extract`
falls back rather than raising there: you asked for an extraction, not for a particular
transport.

### Which providers have it

Each native mode is a different wire format, and they have not all been confirmed
against a live endpoint:

| Mode | Providers | Live-verified? |
|---|---|---|
| `ollama` (`format`) | Ollama | Yes |
| `openai_json_schema` (`response_format`) | OpenAI, Mistral, xAI, Together, Fireworks, OpenRouter, Cerebras, Perplexity | Request path yes, individual hosts no |
| `anthropic_tool` (forced tool call) | Anthropic | **No** |
| `gemini` (`responseSchema`) | Gemini | **No** |
| `none` (prompt path) | Groq, DeepSeek | n/a |

The two unverified modes are the most intricate of the four. See
[Provider verification status](https://github.com/openintelligence-labs/actants/blob/main/docs/PROVIDER_VERIFICATION.md).

### Why Groq and DeepSeek decline it

Both speak the OpenAI wire format, and both are still marked `none`. Speaking a wire
format is not the same as implementing every parameter in it, and the two fail differently:

- **DeepSeek** rejects the request outright. Its `response_format` accepts only `text` and
  `json_object`, not `json_schema`.
- **Groq** accepts `strict` and then ignores it on most models — it is honoured on the
  gpt-oss models only. That is the worse failure, because it **fails open**: the request
  succeeds, the output is unconstrained, and nothing says so.

An extraction that is silently unconstrained while reporting itself as native is exactly
the guarantee this feature exists to provide, so both take the prompt path instead. That
costs some tokens and a possible repair round-trip, and it is honest about what it is.

If Groq ships broader `strict` support, this decision should be revisited — it is a
current-behaviour judgement, not a permanent one.

### `last_schema_plan()`

`last_schema_plan()` is the supported way to tell which path ran:

```python
person = await llm.extract("John is 30 years old.", Person)

plan = llm.last_schema_plan()
print(plan.native)  # True if the schema went on the wire
print(plan.mode)  # the provider's declared capability
print(plan.reason)  # why the native path was declined, or None
```

`mode` is the provider's *capability*, not what ran. A plan can have
`mode="openai_json_schema"` and `native=False` — that means the provider can do it but
this schema could not be expressed in strict form, and `reason` says which construct
caused it.

It returns `None` before the first `extract()` / `extract_stream()` call, and reflects one
client's last call — so read it from the same task that made the call. Two concurrent
extractions on one `LLM` overwrite each other's plan.

This is a method rather than a log line on every extraction, because the answer never
changes for a given provider and schema; logging it would be noise on every call to say
the same thing.

### Streaming extraction

`extract_stream()` yields progressively-complete objects parsed from a text stream, and
uses the native path only where that mode streams as *text*. Anthropic's forced tool call
is excluded: its JSON arrives as tool-call input deltas rather than text deltas, so that
provider streams via the prompt path. `last_schema_plan().reason` says so when it happens.

## OpenAI-compatible providers

Groq, Mistral, xAI, DeepSeek, Together, Fireworks, OpenRouter, Cerebras, and
Perplexity all serve the OpenAI wire format, so they are the same provider pointed
at a different host. Each is registered under its own name with its own API-key
environment variable, and each has an extra that installs the OpenAI SDK:

```bash
pip install 'actants[deepseek]'
```

Point one at a self-hosted gateway or proxy by overriding `base_url`:

```python
from actants import LLM
from actants.llm import DeepSeekProvider

LLM(provider=DeepSeekProvider(api_key="...", base_url="http://localhost:8000/v1"))
```

### Cost tracking on these providers

actants ships prices only for models it has verified. These hosts change model
lineups and prices frequently, so their models are deliberately **unpriced**: their
cost is reported as unknown rather than as `$0.00`, which would read as "this run
was free".

```python
from actants import CostTracker

tracker = CostTracker()
# ... run some completions ...
if tracker.has_untracked_cost:
    print("total is a lower bound; unpriced:", tracker.snapshot()["untracked_models"])
```

Add your own prices by writing into `actants.cost.PRICING`.
