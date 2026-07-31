"""Pydantic AI implementations of the three benchmark tasks.

Uses the first-party ``ollama`` model provider shipped in pydantic-ai 2.x.
The provider requires an explicit base URL (it raises ``UserError`` if
neither ``OLLAMA_BASE_URL`` nor ``base_url=`` is supplied), so every task
builds the model through :func:`_model` rather than the ``"ollama:<name>"``
string shorthand.
"""

from __future__ import annotations

import os

_HOST = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")

# --- task A: one completion -------------------------------------------------
# LOC_A_START
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

BASE_URL = f"{_HOST}/v1"


def _model(model: str) -> OpenAIChatModel:
    return OpenAIChatModel(model, provider=OllamaProvider(base_url=BASE_URL))


async def task_completion(model: str, prompt: str) -> str:
    agent = Agent(_model(model))
    result = await agent.run(prompt)
    return result.output


# LOC_A_END


# --- task B: agent with one tool -------------------------------------------
# LOC_B_START
async def task_tool_agent(model: str, prompt: str) -> str:
    agent = Agent(_model(model))

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"18C and raining in {city}"

    result = await agent.run(prompt)
    return result.output


# LOC_B_END


# --- task C: structured output ---------------------------------------------
# NOTE: pydantic-ai defaults to *tool-call* output mode. Against qwen2.5:7b
# that mode fails outright ("Exceeded maximum output retries"), because the
# small model does not reliably emit the synthetic `final_result` tool call.
# `NativeOutput` selects Ollama's JSON-schema-constrained decoding, which is
# what every other framework in this benchmark uses — so this is the
# apples-to-apples path. The extra import is counted in the LOC table.
# LOC_C_START
from pydantic import BaseModel
from pydantic_ai import NativeOutput


class Person(BaseModel):
    name: str
    age: int
    city: str


async def task_structured(model: str, prompt: str) -> Person:
    agent = Agent(_model(model), output_type=NativeOutput(Person))
    result = await agent.run(prompt)
    return result.output


# LOC_C_END

IMPORT_MODULE = "pydantic_ai"
