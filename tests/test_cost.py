from __future__ import annotations

import pytest

from actants.cost.pricing import (
    PRICING,
    estimate_cost,
    estimate_cost_or_none,
    is_priced,
    lookup_price,
)
from actants.cost.tracker import CostTracker
from actants.llm.base import CompletionResult, TokenUsage


def test_ollama_is_free():
    assert estimate_cost("ollama", "llama3.2", 1000, 1000) == 0.0


def test_ollama_free_is_a_known_price_not_an_unknown_one():
    """Local inference really is $0. That must not look like a missing entry."""
    assert is_priced("ollama", "llama3.2")
    assert estimate_cost_or_none("ollama", "llama3.2", 1000, 1000) == 0.0


def test_openai_gpt4o_mini_pricing():
    # 1M prompt tokens * 0.15 + 1M completion tokens * 0.60 = 0.75
    cost = estimate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-9


# --------------------------------------------------------------------------
# Anthropic: the table listed Opus at (15, 75) — 3x the real price.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "prices"),
    [
        ("claude-opus-5", (5.00, 25.00)),
        ("claude-opus-4-8", (5.00, 25.00)),
        ("claude-opus-4-7", (5.00, 25.00)),
        ("claude-opus-4-6", (5.00, 25.00)),
        ("claude-fable-5", (10.00, 50.00)),
        ("claude-sonnet-5", (3.00, 15.00)),
        ("claude-haiku-4-5", (1.00, 5.00)),
    ],
)
def test_anthropic_current_generation_prices(model: str, prices: tuple[float, float]):
    assert lookup_price("anthropic", model) == prices


def test_opus_tier_is_not_the_stale_15_75():
    """Regression: every Opus entry was 3x too expensive, which silently inflated
    every cost report a user built on this SDK."""
    for model in ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"):
        assert lookup_price("anthropic", model) != (15.00, 75.00)


def test_anthropic_opus_pricing():
    # 1000 prompt * 5/1M + 500 completion * 25/1M
    cost = estimate_cost("anthropic", "claude-opus-5", 1000, 500)
    expected = (1000 / 1_000_000) * 5 + (500 / 1_000_000) * 25
    assert abs(cost - expected) < 1e-9


def test_prefix_matching_still_resolves_dated_model_ids():
    """A dated snapshot must price the same as the alias it extends."""
    assert lookup_price("anthropic", "claude-opus-5-20260101") == (5.00, 25.00)
    assert lookup_price("openai", "gpt-4o-2024-11-20") == (2.50, 10.00)


def test_prefix_matching_prefers_the_longest_match():
    """`claude-haiku-4-5-20251001` must not be captured by a shorter sibling entry."""
    assert lookup_price("anthropic", "claude-haiku-4-5-20251001") == (1.00, 5.00)


# --------------------------------------------------------------------------
# Unknown models: "we don't know" must never render as "free".
# --------------------------------------------------------------------------


def test_unknown_provider_price_is_unknown_not_zero():
    assert lookup_price("mystery", "x") is None
    assert estimate_cost_or_none("mystery", "x", 100, 100) is None
    assert is_priced("mystery", "x") is False


def test_unknown_model_price_is_unknown_not_zero():
    assert lookup_price("openai", "definitely-not-a-model-xyz") is None
    assert estimate_cost_or_none("openai", "definitely-not-a-model-xyz", 100, 100) is None


def test_estimate_cost_still_returns_a_float_for_unknown_models():
    """`cost_usd` is a non-optional float, so the legacy helper keeps its contract —
    but it is a floor, and `untracked_models` is what makes that visible."""
    assert estimate_cost("openai", "definitely-not-a-model-xyz", 100, 100) == 0.0


def test_openai_compatible_providers_are_callable_but_unpriced():
    """We can route to them; we have not verified their prices. Unknown, not free."""
    for provider in ("xai", "deepseek", "together", "fireworks", "openrouter"):
        assert provider in PRICING
        assert lookup_price(provider, "some-model") is None


def test_tracker_flags_unknown_model_spend():
    t = CostTracker()
    t.record(
        CompletionResult(
            content="a",
            model="mystery-model-9000",
            provider="xai",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost_usd=0.0,
        )
    )
    assert t.has_untracked_cost
    assert t.snapshot()["untracked_models"] == ["xai/mystery-model-9000"]


def test_tracker_does_not_flag_a_genuinely_free_model():
    t = CostTracker()
    t.record(
        CompletionResult(
            content="a",
            model="llama3.2",
            provider="ollama",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost_usd=0.0,
        )
    )
    assert not t.has_untracked_cost
    assert t.snapshot()["untracked_models"] == []


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
