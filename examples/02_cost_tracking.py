"""Track cost per tag across multiple calls — useful when an agent has phases.

Run: `python examples/02_cost_tracking.py`
"""

from __future__ import annotations

import asyncio
import json

from agentic_kit import LLM, CostTracker


async def main() -> None:
    tracker = CostTracker()
    llm = LLM(cost_tracker=tracker)

    await llm.complete("Plan 3 search queries for: quantum computing", tag="plan")
    await llm.complete("Summarize: Quantum computing uses qubits...", tag="summarize")
    await llm.complete("Critique this plan: step1 step2", tag="critique")

    print(json.dumps(tracker.snapshot(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
