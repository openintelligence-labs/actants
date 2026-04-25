"""Minimal usage — default Ollama provider, no API key.

Prereqs: `ollama serve` + `ollama pull llama3.2`.

Run: `python examples/01_hello_ollama.py`
"""

from __future__ import annotations

import asyncio

from agentic_kit import LLM


async def main() -> None:
    llm = LLM()
    result = await llm.complete(
        "Explain why local-first AI matters in one paragraph.",
        system="You are concise. No preamble.",
    )
    print(result.content)
    print()
    print(f"model={result.model} tokens={result.usage.total_tokens} cost=${result.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
