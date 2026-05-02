from __future__ import annotations

# Prices in USD per 1M tokens: (input, output). Update as providers change pricing.
# Last verified: 2026-04
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "ollama": {
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
        "claude-opus-4-7": (15.00, 75.00),
        "claude-opus-4-6": (15.00, 75.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-sonnet-4-5": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (0.80, 4.00),
        "claude-haiku-4-5": (0.80, 4.00),
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
}


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a completion. Returns 0.0 for unknown providers/models.

    Matching order:
    1. Exact model name.
    2. Prefix match (e.g. ``claude-haiku-4-5-20251001`` matches the prefix ``claude-haiku-4-5``).
    3. Wildcard ``*`` for providers that are uniformly priced (e.g. Ollama is free).
    """
    provider_table = PRICING.get(provider, {})
    if not provider_table:
        return 0.0
    match = provider_table.get(model)
    if match is None:
        for key, prices in provider_table.items():
            if key == "*":
                continue
            if model.startswith(key):
                match = prices
                break
    if match is None and "*" in provider_table:
        match = provider_table["*"]
    if match is None:
        return 0.0
    in_price, out_price = match
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
