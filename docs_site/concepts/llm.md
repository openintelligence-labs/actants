# LLM and providers

`LLM` is the provider-agnostic gateway. Same API across Ollama, OpenAI, Anthropic,
Gemini, and every major OpenAI-compatible host — Groq, Mistral, xAI, DeepSeek,
Together, Fireworks, OpenRouter, Cerebras, and Perplexity.

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
