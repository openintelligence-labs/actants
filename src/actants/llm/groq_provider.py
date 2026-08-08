from __future__ import annotations

from typing import TYPE_CHECKING

from actants.llm.openai_provider import OpenAIProvider
from actants.llm.structured import NativeSchemaMode

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class GroqProvider(OpenAIProvider):
    """Groq via its OpenAI-compatible endpoint. Requires ``pip install actants[openai]``.

    Groq serves Llama/Mixtral/Qwen etc. at extremely low latency.

    Structured output uses the prompt path: Groq honours ``strict`` only on the gpt-oss
    models and silently ignores it elsewhere, so claiming the native path would report a
    guarantee it does not deliver.
    """

    name = "groq"
    native_schema_mode: NativeSchemaMode = "none"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        # Accepts the full parent signature: dropping ``client`` and ``base_url`` made
        # GroqProvider(client=...) a TypeError on a subclass of a class that accepts it.
        super().__init__(api_key=api_key, client=client, base_url=base_url)
