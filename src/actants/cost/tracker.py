from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actants.llm.base import CompletionResult


@dataclass
class CostTracker:
    """In-memory per-tag cost accumulator. Reset at process start."""

    total_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_tag: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, result: CompletionResult, tag: str | None = None) -> None:
        self.total_usd += result.cost_usd
        self.total_prompt_tokens += result.usage.prompt_tokens
        self.total_completion_tokens += result.usage.completion_tokens
        self.by_model[result.model] += result.cost_usd
        if tag:
            self.by_tag[tag] += result.cost_usd

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "by_tag": {k: round(v, 6) for k, v in self.by_tag.items()},
        }
