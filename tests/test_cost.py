from __future__ import annotations

from agentic_kit.cost.pricing import estimate_cost
from agentic_kit.cost.tracker import CostTracker
from agentic_kit.llm.base import CompletionResult, TokenUsage


def test_ollama_is_free():
    assert estimate_cost("ollama", "llama3.2", 1000, 1000) == 0.0


def test_openai_gpt4o_mini_pricing():
    # 1M prompt tokens * 0.15 + 1M completion tokens * 0.60 = 0.75
    cost = estimate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-9


def test_anthropic_opus_pricing():
    # 1000 prompt * 15/1M + 500 completion * 75/1M
    cost = estimate_cost("anthropic", "claude-opus-4-6", 1000, 500)
    expected = (1000 / 1_000_000) * 15 + (500 / 1_000_000) * 75
    assert abs(cost - expected) < 1e-9


def test_unknown_provider_returns_zero():
    assert estimate_cost("mystery", "x", 100, 100) == 0.0


def test_unknown_model_returns_zero():
    assert estimate_cost("openai", "definitely-not-a-model-xyz", 100, 100) == 0.0


def test_tracker_accumulates():
    t = CostTracker()
    r1 = CompletionResult(
        content="a",
        model="gpt-4o-mini",
        provider="openai",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cost_usd=0.05,
    )
    r2 = CompletionResult(
        content="b",
        model="llama3.2",
        provider="ollama",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
        cost_usd=0.0,
    )
    t.record(r1, tag="search")
    t.record(r2, tag="search")
    snap = t.snapshot()
    assert snap["total_usd"] == 0.05
    assert snap["total_prompt_tokens"] == 300
    assert snap["total_completion_tokens"] == 130
    assert snap["by_model"]["gpt-4o-mini"] == 0.05
    assert snap["by_tag"]["search"] == 0.05
