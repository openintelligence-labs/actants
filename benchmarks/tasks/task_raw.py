"""Raw ollama-python implementations — the "no framework" floor.

This is the control. Any framework that cannot beat these numbers on latency
is charging overhead for its abstractions; the point of the comparison is to
show how much.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")

# --- task A: one completion -------------------------------------------------
# LOC_A_START
from ollama import AsyncClient


async def task_completion(model: str, prompt: str) -> str:
    client = AsyncClient(host=BASE_URL)
    response = await client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# LOC_A_END


# --- task B: agent with one tool -------------------------------------------
# LOC_B_START
import json


async def get_weather(city: str) -> str:
    return f"18C and raining in {city}"


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def task_tool_agent(model: str, prompt: str) -> str:
    client = AsyncClient(host=BASE_URL)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    for _ in range(4):
        response = await client.chat(model=model, messages=messages, tools=[TOOL_SCHEMA])
        message = response["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return message["content"]
        for call in calls:
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            output = await get_weather(**args)
            messages.append({"role": "tool", "content": output, "name": call["function"]["name"]})
    return messages[-1].get("content", "")


# LOC_B_END


# --- task C: structured output ---------------------------------------------
# LOC_C_START
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    city: str


async def task_structured(model: str, prompt: str) -> Person:
    response = await AsyncClient(host=BASE_URL).chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=Person.model_json_schema(),
    )
    return Person.model_validate_json(response["message"]["content"])


# LOC_C_END

IMPORT_MODULE = "ollama"
