"""The documented extras must match the ones pyproject actually declares.

`installation.md` described `all` as "OpenAI + Anthropic + cache + cli" while the
extra had also contained `mcp` for some time, so a reader installed `[all]` and then
installed `[mcp]` again for no reason. Prose describing packaging metadata drifts from
it silently; these tests make that a failure instead.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Extras that exist to name a provider in the "install this extra" error message.
#: Every one resolves to the OpenAI SDK, because that is the client they all speak
#: through — see `actants.llm.openai_compatible`.
_ALIAS_OF_OPENAI = frozenset(
    {
        "groq",
        "mistral",
        "xai",
        "deepseek",
        "together",
        "fireworks",
        "openrouter",
        "cerebras",
        "perplexity",
    }
)

#: Not rolled into `all`: a2a pulls a web server (starlette + uvicorn), which is a
#: heavier thing to install than a client library and is only wanted deliberately.
_EXCLUDED_FROM_ALL = frozenset({"a2a", "dev", "docs"})


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_all_covers_every_extra_except_the_documented_exclusions() -> None:
    extras = _extras()
    everything = set(extras["all"])
    uncovered = {
        name
        for name, deps in extras.items()
        if name not in _EXCLUDED_FROM_ALL and name != "all" and not set(deps) <= everything
    }
    assert not uncovered, (
        f"extras {sorted(uncovered)} are not covered by `all`. Either add them, or add "
        "them to _EXCLUDED_FROM_ALL here and say so in docs_site/installation.md — a "
        "reader who installs [all] and still hits a missing dependency has been misled."
    )


def test_installation_table_does_not_understate_all() -> None:
    """`installation.md`'s `all` row must name every non-alias extra it pulls in."""
    extras = _extras()
    everything = set(extras["all"])
    named = {
        name
        for name, deps in extras.items()
        if name not in {"all", *_EXCLUDED_FROM_ALL}
        and name not in _ALIAS_OF_OPENAI
        and deps  # `gemini` is empty: it rides on httpx, a core dependency
        and set(deps) <= everything
    }
    row = next(
        line
        for line in (REPO_ROOT / "docs_site" / "installation.md")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("| `all`")
    )
    missing = {name for name in named if name not in row}
    assert not missing, (
        f"docs_site/installation.md's `all` row does not mention {sorted(missing)}, "
        f"which `[all]` installs. Row reads: {row}"
    )


def test_every_provider_has_an_extra_to_install() -> None:
    """The missing-extra error tells users to `pip install 'actants[<provider>]'`.

    That instruction only works if the extra exists, so a provider added to the client
    without a matching extra would print advice that fails.
    """
    from actants.llm.client import _PROVIDER_REQUIREMENTS

    extras = _extras()
    missing = {
        extra
        for _, extra in _PROVIDER_REQUIREMENTS.values()
        if extra != "ollama" and extra not in extras
    }
    assert not missing, (
        f"providers advertise extras {sorted(missing)} that pyproject does not define; "
        "`pip install 'actants[...]'` would fail for anyone following the error message."
    )
