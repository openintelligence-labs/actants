"""The public API must be usable from a `mypy --strict` consumer.

`py.typed` promises downstream type checking works. Because `__all__` is built
dynamically from the lazy-import table, the `if TYPE_CHECKING:` block has to use
the explicit `X as X` re-export form — otherwise every `from actants import LLM`
fails strict checking with "does not explicitly export attribute". These tests
pin that down so the promise cannot silently regress.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_PATH = REPO_ROOT / "src" / "actants" / "__init__.py"


def _type_checking_import_aliases() -> list[ast.alias]:
    tree = ast.parse(INIT_PATH.read_text(encoding="utf-8"))
    aliases: list[ast.alias] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not is_type_checking:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                aliases.extend(child.names)
    return aliases


def test_type_checking_block_uses_explicit_reexports():
    aliases = _type_checking_import_aliases()
    assert aliases, "no TYPE_CHECKING imports found in actants/__init__.py"
    bad = [a.name for a in aliases if a.asname != a.name]
    assert not bad, (
        "these TYPE_CHECKING imports are not explicit re-exports and will break "
        f"`mypy --strict` for consumers: {bad}. Use `from x import Y as Y`."
    )


def test_every_public_symbol_is_reexported_for_type_checkers():
    """Everything in __all__ must also appear in the TYPE_CHECKING block."""
    import actants

    reexported = {a.asname or a.name for a in _type_checking_import_aliases()}
    runtime = {n for n in actants.__all__ if n != "__version__"}
    missing = sorted(runtime - reexported)
    assert not missing, (
        f"public symbols missing from the TYPE_CHECKING re-export block: {missing}. "
        "Type checkers and IDEs will not see them."
    )


def test_every_public_symbol_resolves_at_runtime():
    import actants

    unresolvable = []
    for name in actants.__all__:
        if name == "__version__":
            continue
        try:
            getattr(actants, name)
        except Exception as exc:  # noqa: BLE001 - reporting all failures at once
            unresolvable.append(f"{name} ({type(exc).__name__}: {exc})")
    assert not unresolvable, f"__all__ names that do not resolve: {unresolvable}"


CONSUMER = textwrap.dedent(
    """
    from __future__ import annotations

    import asyncio

    from actants import Agent, AgentResult, ChatMessage, LLM, LLMSettings, ToolRegistry


    async def add(a: int, b: int) -> int:
        return a + b


    async def main() -> str:
        tools = ToolRegistry()
        tools.register_function("add", "Add two integers", add)
        agent: Agent = Agent(llm=LLM(settings=LLMSettings()), tools=tools)
        result: AgentResult = await agent.run("what is 2+2")
        messages: list[ChatMessage] = result.messages
        assert messages is not None
        return result.content


    if __name__ == "__main__":
        asyncio.run(main())
    """
)


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy not installed")
def test_consumer_script_passes_mypy_strict(tmp_path: Path):
    """A `mypy --strict` consumer of the public API must type-check cleanly.

    This is the test that would have caught `py.typed` being effectively broken:
    before the explicit re-export fix, every `from actants import ...` produced
    "does not explicitly export attribute".
    """
    script = tmp_path / "consumer.py"
    script.write_text(CONSUMER, encoding="utf-8")
    proc = subprocess.run(
        [
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy"),
            str(script),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode == 2 and "INTERNAL ERROR" in proc.stderr:
        pytest.skip(f"mypy crashed internally (not an actants failure): {proc.stderr[-300:]}")
    assert proc.returncode == 0, (
        f"mypy --strict failed on a consumer of the public API:\n{proc.stdout}\n{proc.stderr}"
    )


SUBPACKAGE_CONSUMER = textwrap.dedent(
    """
    from __future__ import annotations

    from actants.cache import CacheRequest, InMemoryCache, RequestCacheBackend, SqliteVecCache
    from actants.llm import AnthropicProvider, OllamaProvider, OpenAIProvider


    def names() -> list[str]:
        return [
            InMemoryCache.__name__,
            SqliteVecCache.__name__,
            CacheRequest.__name__,
            RequestCacheBackend.__name__,
            OllamaProvider.__name__,
            OpenAIProvider.__name__,
            AnthropicProvider.__name__,
        ]
    """
)


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy not installed")
def test_subpackage_imports_pass_mypy_strict(tmp_path: Path):
    """Subpackage imports must type-check too, not just the top-level ones.

    `actants.cache` and `actants.llm` resolve some names through a module-level
    `__getattr__`. Without a matching `TYPE_CHECKING` re-export block, every
    `from actants.cache import SqliteVecCache` failed strict checking with
    "does not explicitly export attribute" — while `InMemoryCache`, imported
    normally in the same module, worked. That asymmetry is the bug.
    """
    script = tmp_path / "sub_consumer.py"
    script.write_text(SUBPACKAGE_CONSUMER, encoding="utf-8")
    proc = subprocess.run(
        [
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy"),
            str(script),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode == 2 and "INTERNAL ERROR" in proc.stderr:
        pytest.skip(f"mypy crashed internally (not an actants failure): {proc.stderr[-300:]}")
    assert proc.returncode == 0, (
        f"mypy --strict failed on subpackage imports:\n{proc.stdout}\n{proc.stderr}"
    )


def test_lazy_subpackage_all_entries_resolve():
    """Every name in a lazy subpackage's __all__ must actually resolve."""
    import importlib

    for mod_name in ("actants.cache", "actants.llm"):
        module = importlib.import_module(mod_name)
        unresolvable = []
        for name in module.__all__:
            try:
                getattr(module, name)
            except Exception as exc:  # noqa: BLE001 - report all at once
                unresolvable.append(f"{mod_name}.{name} ({type(exc).__name__}: {exc})")
        assert not unresolvable, f"__all__ names that do not resolve: {unresolvable}"
