from __future__ import annotations

from agentic_kit.llm.openai_provider import OpenAIProvider


class MistralProvider(OpenAIProvider):
    """Mistral via its OpenAI-compatible endpoint. Requires ``pip install agentic-kit[openai]``."""

    name = "mistral"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url="https://api.mistral.ai/v1")
