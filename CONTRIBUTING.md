# Contributing to actants

Thanks for your interest! This package is the shared backbone of the Open Intelligence Labs ecosystem, so changes here affect every downstream project.

## Dev setup

```bash
git clone https://github.com/openintelligence-labs/actants
cd actants
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,openai,anthropic,cache]'
pytest tests/
```

## Before opening a PR

- `ruff check .` passes
- `ruff format --check .` passes
- `pytest tests/` passes
- `mypy --strict src/` passes — CI checks the whole tree, not just the entry point
- New public functions have docstrings
- Tests cover the new behavior

## Principles

- **Local-first**: default to Ollama, no API key required
- **Privacy**: no telemetry, no phone-home
- **Async-only**: every I/O path uses `async`/`await`
- **Typed**: full type hints, pydantic models for data

Come chat on [Discord](https://discord.gg/openintelligence-labs).
