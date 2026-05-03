"""Watch a structured report take shape as the model streams JSON.

Run: `python examples/08_extract_stream.py`
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from actants import LLM


class TripPlan(BaseModel):
    destination: str
    days: int
    highlights: list[str]
    estimated_budget_usd: float


async def main() -> None:
    llm = LLM()
    async for partial in llm.extract_stream(
        "Plan a 5-day trip to Kyoto in spring. Be specific.",
        TripPlan,
    ):
        print("--- partial ---")
        print(partial.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
