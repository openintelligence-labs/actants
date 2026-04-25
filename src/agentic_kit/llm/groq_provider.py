from __future__ import annotations

from agentic_kit.llm.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq via its OpenAI-compatible endpoint. Requires ``pip install agentic-kit[openai]``.

    Groq serves Llama/Mixtral/Qwen etc. at extremely low latency.
    """

    name = "groq"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url="https://api.groq.com/openai/v1")
