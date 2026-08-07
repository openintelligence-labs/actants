"""Per-model token prices, and honest cost estimation.

The table below is the *only* place actants asserts what a token costs. Every entry
is a published list price in USD per 1M tokens, ``(input, output)``. An entry that
cannot be verified is not added: a wrong price in a cost-tracking SDK is worse than
no price, because a wrong number is still believed.

The same rule shapes :func:`lookup_price`. A model that is not in the table has an
*unknown* cost, not a zero cost — see :func:`estimate_cost` for how that is surfaced.
"""

from __future__ import annotations

# Prices in USD per 1M tokens: (input, output). Update as providers change pricing.
# Last verified: 2026-08 (anthropic), 2026-04 (everything else).
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "ollama": {
        # Local inference. Free is a real price here, not a missing one.
        "*": (0.0, 0.0),
    },
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "o1": (15.00, 60.00),
        "o1-mini": (3.00, 12.00),
        "o3": (10.00, 40.00),
        "o3-mini": (1.10, 4.40),
    },
    "anthropic": {
        # Current generation. Opus-tier is $5/$25 per 1M — the table previously
        # carried the older $15/$75, overstating every Opus call by 3x.
        "claude-opus-5": (5.00, 25.00),
        "claude-opus-4-8": (5.00, 25.00),
        "claude-opus-4-7": (5.00, 25.00),
        "claude-opus-4-6": (5.00, 25.00),
        "claude-fable-5": (10.00, 50.00),
        "claude-sonnet-5": (3.00, 15.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-sonnet-4-5": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        "claude-haiku-4-5": (1.00, 5.00),
    },
    "gemini": {
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.075, 0.30),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
    },
    "groq": {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "qwen-qwq-32b": (0.29, 0.39),
    },
    "mistral": {
        "mistral-large-latest": (2.00, 6.00),
        "mistral-small-latest": (0.20, 0.60),
        "codestral-latest": (0.30, 0.90),
        "pixtral-large-latest": (2.00, 6.00),
    },
    # Providers below reach the wire through the OpenAI-compatible provider. They are
    # listed here with no entries because actants can call them but cannot price them:
    # an empty table means every model resolves as *unknown*, which is the honest
    # answer, rather than silently falling through to 0.0.
    "xai": {},
    "deepseek": {},
    "together": {},
    "fireworks": {},
    "openrouter": {},
    "cerebras": {},
    "perplexity": {},
}


def lookup_price(provider: str, model: str) -> tuple[float, float] | None:
    """Return ``(input, output)`` USD-per-1M prices, or ``None`` if unknown.

    This is the honest primitive: a provider or model actants has no published price
    for returns ``None``, which a caller can render as "unknown" rather than "$0.00".

    Matching order:
    1. Exact model name.
    2. Longest prefix match (e.g. ``claude-opus-5-20260101`` matches ``claude-opus-5``).
       Longest wins so that ``claude-haiku-4-5-20251001`` cannot be captured by a
       shorter, differently-priced sibling entry.
    3. Wildcard ``*`` for providers that are uniformly priced (Ollama is free).
    """
    provider_table = PRICING.get(provider)
    if not provider_table:
        return None
    match = provider_table.get(model)
    if match is not None:
        return match
    best: str | None = None
    for key in provider_table:
        if key == "*":
            continue
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return provider_table[best]
    return provider_table.get("*")


def estimate_cost_or_none(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Estimate USD cost, or ``None`` when the model's price is unknown.

    Prefer this over :func:`estimate_cost` anywhere the difference between "free" and
    "we don't know" is visible to a human — a spend report that prints ``$0.00`` for an
    unpriced model reads as "this was free", which is the failure this function exists
    to prevent.
    """
    prices = lookup_price(provider, model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def is_priced(provider: str, model: str) -> bool:
    """Whether actants has a published price for this provider/model pair."""
    return lookup_price(provider, model) is not None


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a completion, returning ``0.0`` when the price is unknown.

    ``0.0`` here is a *floor*, not a claim that the call was free. The field it feeds
    (:attr:`~actants.llm.base.CompletionResult.cost_usd`) is a non-optional float, so
    an unpriced model has to be some number; this keeps totals from being inflated by
    a guess. Callers that need to tell "free" from "unknown" apart must use
    :func:`estimate_cost_or_none` or :func:`is_priced` — and
    :attr:`~actants.cost.tracker.CostTracker.untracked_models` records every model
    that took this path, so an unknown model is visible in the report instead of
    disappearing into a $0.00 line.
    """
    return estimate_cost_or_none(provider, model, prompt_tokens, completion_tokens) or 0.0
