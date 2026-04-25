"""Same prompt, six providers — showcases the unified API.

Run:
  `OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \\
   GROQ_API_KEY=... MISTRAL_API_KEY=... python examples/09_provider_zoo.py`

Any missing key is skipped.
"""

from __future__ import annotations

import asyncio
import os

from agentic_kit import LLM, OllamaProvider


async def run(name: str, provider, model: str) -> None:
    try:
        llm = LLM(provider=provider, model=model)
        r = await llm.complete("One sentence: why does local-first AI matter?")
        print(f"[{name:>9} / {model}] {r.content.strip()}")
    except Exception as exc:
        print(f"[{name:>9}] skipped: {exc}")


async def main() -> None:
    await run("ollama", OllamaProvider(), "llama3.2")

    if os.environ.get("OPENAI_API_KEY"):
        from agentic_kit.llm.openai_provider import OpenAIProvider

        await run("openai", OpenAIProvider(), "gpt-4o-mini")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from agentic_kit.llm.anthropic_provider import AnthropicProvider

        await run("anthropic", AnthropicProvider(), "claude-haiku-4-5-20251001")

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        from agentic_kit.llm.gemini_provider import GeminiProvider

        await run("gemini", GeminiProvider(), "gemini-2.5-flash")

    if os.environ.get("GROQ_API_KEY"):
        from agentic_kit.llm.groq_provider import GroqProvider

        await run(
            "groq",
            GroqProvider(api_key=os.environ["GROQ_API_KEY"]),
            "llama-3.3-70b-versatile",
        )

    if os.environ.get("MISTRAL_API_KEY"):
        from agentic_kit.llm.mistral_provider import MistralProvider

        await run(
            "mistral",
            MistralProvider(api_key=os.environ["MISTRAL_API_KEY"]),
            "mistral-small-latest",
        )


if __name__ == "__main__":
    asyncio.run(main())
