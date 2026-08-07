from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from actants.cost.pricing import is_priced

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
    #: Models recorded that actants has no published price for, as ``"provider/model"``.
    #: Their spend is counted as 0.0 in the totals above, so a non-empty set means the
    #: totals are a *lower bound* — not a complete bill. Reported by :meth:`snapshot`
    #: as ``untracked_models`` so an unpriced model is visible rather than silently
    #: contributing $0.00 to a total that looks authoritative.
    untracked_models: set[str] = field(default_factory=set)

    def record(self, result: CompletionResult, tag: str | None = None) -> None:
        self.total_usd += result.cost_usd
        self.total_prompt_tokens += result.usage.prompt_tokens
        self.total_completion_tokens += result.usage.completion_tokens
        self.by_model[result.model] += result.cost_usd
        if tag:
            self.by_tag[tag] += result.cost_usd
        if not is_priced(result.provider, result.model):
            self.untracked_models.add(f"{result.provider}/{result.model}")

    @property
    def has_untracked_cost(self) -> bool:
        """True when at least one recorded call had no known price.

        When this is true, :attr:`total_usd` understates real spend. Surface it next to
        the total wherever the total is shown to a person.
        """
        return bool(self.untracked_models)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "by_tag": {k: round(v, 6) for k, v in self.by_tag.items()},
            # Sorted so the snapshot is stable enough to assert on and diff.
            "untracked_models": sorted(self.untracked_models),
        }
