# LLM and providers

`LLM` is the provider-agnostic gateway. Same API across Ollama, OpenAI, Anthropic, Gemini, Groq, Mistral.

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
