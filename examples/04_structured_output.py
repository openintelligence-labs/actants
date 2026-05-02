"""Extract a typed pydantic object from any provider — works with Ollama too.

Run: `python examples/04_structured_output.py`
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from actants import LLM


class IssueReport(BaseModel):
    title: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    components: list[str]
    summary: str


async def main() -> None:
    llm = LLM()
    issue = await llm.extract(
        (
            "Extract a structured report from this bug text:\n\n"
            "The checkout page crashes when a coupon with 0% discount is applied. "
            "Affects web and mobile. Blocks release."
        ),
        IssueReport,
    )
    print(issue.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
