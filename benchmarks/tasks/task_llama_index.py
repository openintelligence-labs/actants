"""LlamaIndex implementations of the three benchmark tasks.

Uses ``llama-index-core`` plus the ``llama-index-llms-ollama`` integration.
The tool task uses ``FunctionAgent`` from ``llama_index.core.agent.workflow``,
the current documented agent entrypoint in llama-index-core 0.14.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")

# --- task A: one completion -------------------------------------------------
# LOC_A_START
from llama_index.llms.ollama import Ollama


async def task_completion(model: str, prompt: str) -> str:
    llm = Ollama(model=model, base_url=BASE_URL, request_timeout=600.0)
    result = await llm.acomplete(prompt)
    return result.text


# LOC_A_END


# --- task B: agent with one tool -------------------------------------------
# LOC_B_START
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool


async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and raining in {city}"


async def task_tool_agent(model: str, prompt: str) -> str:
    agent = FunctionAgent(
        tools=[FunctionTool.from_defaults(async_fn=get_weather)],
        llm=Ollama(model=model, base_url=BASE_URL, request_timeout=600.0),
    )
    return str(await agent.run(prompt))


# LOC_B_END


# --- task C: structured output ---------------------------------------------
# LOC_C_START
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    city: str


async def task_structured(model: str, prompt: str) -> Person:
    llm = Ollama(model=model, base_url=BASE_URL, request_timeout=600.0)
    return await llm.astructured_predict(Person, PromptTemplate("{text}"), text=prompt)


# LOC_C_END

IMPORT_MODULE = "llama_index.core"
