from __future__ import annotations

from typing import TYPE_CHECKING

from actants.llm.openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class MistralProvider(OpenAIProvider):
    """Mistral via its OpenAI-compatible endpoint. Requires ``pip install actants[openai]``."""

    name = "mistral"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str = "https://api.mistral.ai/v1",
    ) -> None:
        # Accepts the full parent signature; see GroqProvider for why.
        super().__init__(api_key=api_key, client=client, base_url=base_url)
