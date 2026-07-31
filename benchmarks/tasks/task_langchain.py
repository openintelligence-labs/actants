"""LangChain implementations of the three benchmark tasks.

Uses ``langchain-ollama`` for the model and ``langchain.agents.create_agent``
(the LangGraph-backed agent constructor that is the documented path in
LangChain 1.x) for the tool task.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")

# --- task A: one completion -------------------------------------------------
# LOC_A_START
from langchain_ollama import ChatOllama


async def task_completion(model: str, prompt: str) -> str:
    llm = ChatOllama(model=model, base_url=BASE_URL)
    result = await llm.ainvoke(prompt)
    return result.content


# LOC_A_END


# --- task B: agent with one tool -------------------------------------------
# LOC_B_START
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and raining in {city}"


async def task_tool_agent(model: str, prompt: str) -> str:
    agent = create_agent(ChatOllama(model=model, base_url=BASE_URL), tools=[get_weather])
    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


# LOC_B_END


# --- task C: structured output ---------------------------------------------
# LOC_C_START
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    city: str


async def task_structured(model: str, prompt: str) -> Person:
    llm = ChatOllama(model=model, base_url=BASE_URL).with_structured_output(Person)
    return await llm.ainvoke(prompt)


# LOC_C_END

IMPORT_MODULE = "langchain"
