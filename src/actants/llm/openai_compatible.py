"""Providers that speak the OpenAI wire format at a different base URL.

xAI, DeepSeek, Together, Fireworks, OpenRouter, Cerebras, and Perplexity all expose
an OpenAI-compatible ``/chat/completions`` endpoint. There is nothing for actants to
translate, so each one is `OpenAIProvider` with a
``name`` and a ``base_url`` — the same shape ``GroqProvider`` and ``MistralProvider``
already had, which is why those two are declared in the table here as well and their
original modules now re-export from it.

Writing nine near-identical class bodies by hand would mean nine places for the
request path to drift apart. They are built from one table instead, and
`openai_compatible_provider` builds the class so each one is still a real,
importable, subclassable type with the parent's full signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from actants.llm.openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI

#: provider name -> (base URL, one-line description used in the generated docstring).
#:
#: Adding a row here is the whole cost of supporting another OpenAI-compatible host.
#: A row must NOT be added for a provider whose request or response shape differs from
#: OpenAI's — that provider needs a real class, because the point of this module is
#: that the wire format is genuinely identical.
OPENAI_COMPATIBLE_PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "Groq — Llama/Mixtral/Qwen at very low latency"),
    "mistral": ("https://api.mistral.ai/v1", "Mistral — Mistral Large, Codestral, Pixtral"),
    "xai": ("https://api.x.ai/v1", "xAI — the Grok models"),
    "deepseek": ("https://api.deepseek.com/v1", "DeepSeek — deepseek-chat and deepseek-reasoner"),
    "together": ("https://api.together.xyz/v1", "Together AI — a large open-model catalogue"),
    "fireworks": (
        "https://api.fireworks.ai/inference/v1",
        "Fireworks AI — fast serving of open models",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OpenRouter — one endpoint routing to many upstream providers",
    ),
    "cerebras": ("https://api.cerebras.ai/v1", "Cerebras — open models on Cerebras hardware"),
    "perplexity": ("https://api.perplexity.ai", "Perplexity — the Sonar search-grounded models"),
}

#: Generated class name overrides. ``str.title()`` would produce ``XaiProvider`` and
#: ``DeepseekProvider``, which is not what a user types at the import site.
_CLASS_NAMES: dict[str, str] = {
    "xai": "XAIProvider",
    "deepseek": "DeepSeekProvider",
    "openrouter": "OpenRouterProvider",
}


def openai_compatible_provider(name: str, base_url: str, description: str) -> type[OpenAIProvider]:
    """Build an `OpenAIProvider` subclass pinned to ``base_url``.

    The generated class accepts the *full* parent signature — ``api_key``, ``client``,
    and ``base_url`` — because narrowing a public subclass's signature makes
    ``Provider(client=...)`` a TypeError on a parameter the parent accepts. That was a
    real bug in ``GroqProvider``; generating the class keeps it from coming back one
    subclass at a time.
    """
    default_base_url = base_url

    def __init__(  # noqa: N807 - this becomes the generated class's __init__
        self: OpenAIProvider,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str = default_base_url,
    ) -> None:
        OpenAIProvider.__init__(self, api_key=api_key, client=client, base_url=base_url)

    cls_name = _CLASS_NAMES.get(name, f"{name.title()}Provider")
    cls = type(
        cls_name,
        (OpenAIProvider,),
        {
            "__doc__": (
                f"{description}, via its OpenAI-compatible endpoint.\n\n"
                f"Requires ``pip install 'actants[{name}]'`` (the OpenAI SDK).\n"
                f"Defaults to ``base_url={default_base_url!r}``."
            ),
            "__module__": __name__,
            "__init__": __init__,
            "name": name,
        },
    )
    return cls


XAIProvider = openai_compatible_provider("xai", *OPENAI_COMPATIBLE_PROVIDERS["xai"])
DeepSeekProvider = openai_compatible_provider("deepseek", *OPENAI_COMPATIBLE_PROVIDERS["deepseek"])
TogetherProvider = openai_compatible_provider("together", *OPENAI_COMPATIBLE_PROVIDERS["together"])
FireworksProvider = openai_compatible_provider(
    "fireworks", *OPENAI_COMPATIBLE_PROVIDERS["fireworks"]
)
OpenRouterProvider = openai_compatible_provider(
    "openrouter", *OPENAI_COMPATIBLE_PROVIDERS["openrouter"]
)
CerebrasProvider = openai_compatible_provider("cerebras", *OPENAI_COMPATIBLE_PROVIDERS["cerebras"])
PerplexityProvider = openai_compatible_provider(
    "perplexity", *OPENAI_COMPATIBLE_PROVIDERS["perplexity"]
)

__all__ = [
    "OPENAI_COMPATIBLE_PROVIDERS",
    "CerebrasProvider",
    "DeepSeekProvider",
    "FireworksProvider",
    "OpenRouterProvider",
    "PerplexityProvider",
    "TogetherProvider",
    "XAIProvider",
    "openai_compatible_provider",
]
