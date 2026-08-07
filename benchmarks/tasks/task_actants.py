"""actants implementations of the three benchmark tasks.

Each ``task_*`` coroutine performs exactly one unit of work and returns the
value the runner asserts on.

``BASE_URL`` is benchmark scaffolding: the runner points every framework at a
recording proxy so model time can be separated from framework overhead. Every
task file in this directory has an equivalent one-line shim and none of them
is charged for it in the LOC count -- a real user omits it and gets the
``localhost:11434`` default.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")

# --- task A: one completion -------------------------------------------------
# LOC_A_START
from actants import LLM, LLMSettings


async def task_completion(model: str, prompt: str) -> str:
    llm = LLM(settings=LLMSettings(model=model, base_url=BASE_URL))
    result = await llm.complete(prompt)
    return result.content


# LOC_A_END


# --- task B: agent with one tool -------------------------------------------
# LOC_B_START
from actants import ToolRegistry


async def get_weather(city: str) -> str:
    return f"18C and raining in {city}"


def _build_tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register_function("get_weather", "Get the current weather for a city.", get_weather)
    return tools


async def task_tool_agent(model: str, prompt: str) -> str:
    llm = LLM(settings=LLMSettings(model=model, base_url=BASE_URL))
    result = await llm.run_agent(prompt, tools=_build_tools(), max_steps=4)
    return result.content


# LOC_B_END


# --- task C: structured output ---------------------------------------------
# LOC_C_START
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    city: str


async def task_structured(model: str, prompt: str) -> Person:
    llm = LLM(settings=LLMSettings(model=model, base_url=BASE_URL))
    return await llm.extract(prompt, Person)


# LOC_C_END

IMPORT_MODULE = "actants"
