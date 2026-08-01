# Configuration

actants reads configuration from three places, in this order:

1. **Constructor arguments** — `LLM(provider=..., model=...)`, `Agent(...)`, etc.
2. **Environment variables** — prefixed `ACTANTS_*` for the LLM client, or
   per-provider env vars for API keys.
3. **`.env` files** — loaded automatically by pydantic-settings when present
   in the working directory.

Constructor args win over env vars, env vars win over `.env`.

## LLM client environment variables

| Variable | Default | Description |
|---|---|---|
| `ACTANTS_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `gemini`, `groq`, `mistral` |
| `ACTANTS_MODEL` | `llama3.2` | Model id passed to the provider |
| `ACTANTS_TEMPERATURE` | `0.7` | Sampling temperature |
| `ACTANTS_MAX_TOKENS` | `None` | Max tokens to generate |
| `ACTANTS_BASE_URL` | provider-specific | Override the API base URL (self-hosted, Groq, OpenAI-compatible servers) |

## Provider API keys

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Ollama | _none_ — local |

## Embeddings env vars

| Variable | Default | Description |
|---|---|---|
| `ACTANTS_EMBED_PROVIDER` | `ollama` | Embedding provider name |
| `ACTANTS_EMBED_MODEL` | `nomic-embed-text` | Embedding model id |
| `ACTANTS_EMBED_BASE_URL` | `http://localhost:11434` | Provider URL |

## App-level settings

For your own app's settings, subclass `AppSettings`:

```python
from actants.config import AppSettings


class MySettings(AppSettings):
    model_config = AppSettings.config_for("deepdive")
    search_provider: str = "ddg"
    max_results: int = 10


s = MySettings()  # reads from .env + DEEPDIVE_* env vars
print(s.search_provider, s.max_results)
```

`config_for(app_name)` derives `env_prefix` from `app_name.upper()` (so
`deepdive` → `DEEPDIVE_*`).

## Per-user paths (XDG-aware)

For databases, caches, and config files, use the platform-aware helpers:

```python
from actants.config import app_config_dir, app_data_dir, app_cache_dir

cfg = app_config_dir("deepdive")  # ~/Library/Application Support/deepdive   (macOS)
# ~/.config/deepdive                       (Linux)
# %APPDATA%/deepdive                       (Windows)
data = app_data_dir("deepdive")  # databases, models, large state
cache = app_cache_dir("deepdive")  # regenerable artifacts
```

Each helper creates the directory by default (`create=False` to skip).

## Telemetry

actants ships **zero outbound telemetry**. There is no `--analytics-opt-out`
flag because there is nothing to opt out of. CI asserts the bare-import path
makes no network calls.

If you set up [OpenTelemetry](reference/otel.md), spans go to whatever OTLP
collector you configured — never to us.
