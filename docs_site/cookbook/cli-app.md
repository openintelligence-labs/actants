# Cookbook — Stateful CLI app

A short Click + Rich CLI that wraps an Agent, persists settings to the user's
config dir, and writes JSONL logs of every run.

```python
# my_app.py
from __future__ import annotations

import asyncio

from actants import (
    Agent,
    AppSettings,
    JsonlAppender,
    LLM,
    app_config_dir,
    app_data_dir,
)
from actants.cli import common_options, console, error, make_app, success


class MyAppSettings(AppSettings):
    model_config = AppSettings.config_for("myapp")
    model: str = "llama3.2"
    system: str = "You are a concise assistant."


app = make_app("myapp", help="A tiny stateful CLI agent")


@app.command()
@common_options
@click.option("--prompt", required=True, help="What to ask")
def ask(prompt: str) -> None:
    settings = MyAppSettings()
    log_path = app_data_dir("myapp") / "runs.jsonl"

    async def run() -> None:
        agent = Agent(llm=LLM(model=settings.model))
        result = await agent.run(prompt)
        console.print(result.content)
        with JsonlAppender(log_path) as out:
            out.write(
                {
                    "model": result.final.model,
                    "prompt": prompt,
                    "answer": result.content,
                    "tokens": result.final.usage.total_tokens,
                }
            )

    try:
        asyncio.run(run())
    except Exception as exc:
        error(f"agent failed: {exc}")
    success(f"logged to {log_path}")


if __name__ == "__main__":
    app()
```

Run:

```bash
pip install 'actants[cli]'
python my_app.py ask --prompt "What is local-first software?"
python my_app.py ask --prompt "Why does it matter?" --debug
python my_app.py ask --prompt "Give me bullets" --json
```

## What this demonstrates

- **Per-app settings** with `AppSettings.config_for("myapp")` — env vars are
  picked up automatically (`MYAPP_MODEL=...`, `MYAPP_SYSTEM=...`).
- **XDG paths** — log file lives in the platform's data dir, no hard-coded paths.
- **Common flags** — `--debug`, `--quiet`, `--json`, `--log-format` are wired
  by `@common_options`.
- **Append-only logs** — durable per-record flushes; safe across crashes.
