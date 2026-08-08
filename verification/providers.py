"""The per-provider table the harness runs against.

One row per provider actants supports. The model is always the cheapest one that
provider sells, because this harness exists to check wire formats, not model quality —
a 20-token answer from the cheapest model exercises exactly the same request path as an
expensive one.

``expected_schema_mode`` is what the provider *declares*; the harness asserts the plan
actually taken matches it, which is how a native path that silently fell back to the
prompt path gets caught.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from actants.llm.client import _PROVIDER_REQUIREMENTS


@dataclass(frozen=True)
class ProviderTarget:
    name: str
    model: str
    #: Environment variable holding the key. ``None`` for Ollama, which needs none.
    env_var: str | None
    #: What ``native_schema_mode`` this provider should report on the wire.
    expected_schema_mode: str
    #: Rough USD per 1M tokens, input+output averaged, purely for the pre-run estimate.
    approx_price_per_1m: float
    #: Base URL override. Set only for the local OpenAI-compatible probe below.
    base_url: str | None = None

    def key_present(self) -> bool:
        return self.env_var is None or bool(os.environ.get(self.env_var))

    @property
    def free(self) -> bool:
        return self.env_var is None


TARGETS: tuple[ProviderTarget, ...] = (
    ProviderTarget("ollama", "qwen2.5:7b", None, "ollama", 0.0),
    ProviderTarget("openai", "gpt-4o-mini", "OPENAI_API_KEY", "openai_json_schema", 0.75),
    ProviderTarget("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", "anthropic_tool", 6.0),
    ProviderTarget("gemini", "gemini-2.5-flash", "GEMINI_API_KEY", "gemini", 0.38),
    ProviderTarget("groq", "llama-3.1-8b-instant", "GROQ_API_KEY", "none", 0.13),
    ProviderTarget(
        "mistral", "mistral-small-latest", "MISTRAL_API_KEY", "openai_json_schema", 0.80
    ),
    ProviderTarget("xai", "grok-3-mini", "XAI_API_KEY", "openai_json_schema", 0.80),
    ProviderTarget("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", "none", 1.10),
    ProviderTarget(
        "together",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "TOGETHER_API_KEY",
        "openai_json_schema",
        0.88,
    ),
    ProviderTarget(
        "fireworks",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "FIREWORKS_API_KEY",
        "openai_json_schema",
        0.20,
    ),
    ProviderTarget(
        "openrouter",
        "meta-llama/llama-3.1-8b-instruct",
        "OPENROUTER_API_KEY",
        "openai_json_schema",
        0.05,
    ),
    ProviderTarget("cerebras", "llama3.1-8b", "CEREBRAS_API_KEY", "openai_json_schema", 0.10),
    ProviderTarget("perplexity", "sonar", "PERPLEXITY_API_KEY", "openai_json_schema", 1.00),
)

#: The name of the local probe target. Not an actants provider — see below.
OPENAI_COMPAT_PROBE = "openai-compatible (local probe)"

#: Ollama also serves an OpenAI-compatible endpoint at ``/v1``, which lets the *class*
#: nine paid providers are generated from be exercised against a real server for free.
#:
#: This does not verify any of those nine hosts: each may reject a parameter the others
#: accept, and ``NO_NATIVE_SCHEMA`` exists precisely because two of them do. What it does
#: verify is the shared request path — the strict-schema rewriter, the streaming tool-call
#: assembler, ``stream_options`` usage reporting — against something that answers for
#: real rather than a mock built from the same assumptions as the code.
COMPAT_PROBE = ProviderTarget(
    OPENAI_COMPAT_PROBE,
    "qwen2.5:7b",
    None,
    "openai_json_schema",
    0.0,
    base_url="http://localhost:11434/v1",
)


def _assert_table_covers_actants() -> None:
    """Fail loudly if actants grows a provider this harness does not know about.

    The point of the harness is to say honestly which providers are verified; a provider
    missing from this table would be silently unverified *and* silently unreported.
    """
    known = set(_PROVIDER_REQUIREMENTS)
    covered = {t.name for t in TARGETS}
    missing = known - covered
    if missing:
        raise RuntimeError(
            f"actants supports provider(s) {sorted(missing)} with no row in verification/"
            "providers.py. Add a ProviderTarget so they are either verified or explicitly "
            "reported as skipped."
        )


_assert_table_covers_actants()
