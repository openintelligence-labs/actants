"""Resilience: local-first with a cloud fallback + exponential-backoff retry.

Run: `OPENAI_API_KEY=sk-... python examples/05_fallback_and_retry.py`
"""

from __future__ import annotations

import asyncio
import os

from actants import (
    LLM,
    FallbackProvider,
    OllamaProvider,
    RetryPolicy,
)


async def main() -> None:
    providers: list = [(OllamaProvider(), "llama3.2")]
    if os.environ.get("OPENAI_API_KEY"):
        from actants.llm.openai_provider import OpenAIProvider

        providers.append((OpenAIProvider(), "gpt-4o-mini"))

    llm = LLM(
        provider=FallbackProvider(providers),
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.5),
    )

    result = await llm.complete("One sentence on resilience in distributed systems.")
    print(f"[{result.provider}/{result.model}] {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
