"""Every Python snippet in README.md and docs_site/ must be real code.

This is the mechanism that keeps documentation from drifting away from the API.
Three levels of checking, cheapest first:

1. **Compile** — every snippet must parse as valid Python (catches truncated or
   pseudo-code blocks).
2. **Resolve** — every ``actants`` symbol a snippet imports must actually exist,
   and every ``actants`` attribute/keyword-argument it uses must match the real
   signature (catches renamed kwargs and removed classes — the class of bug that
   produced four broken README snippets).
3. **Execute** — snippets tagged as offline-safe are run for real.

Snippets that need network, a running server, or an API key are skipped for
execution but still compiled and resolved. Mark a block ``<!-- docs-test: skip -->``
to exclude it entirely (use sparingly, and say why).
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATHS = sorted(
    [REPO_ROOT / "README.md", *(REPO_ROOT / "docs_site").rglob("*.md")],
)

FENCE_RE = re.compile(
    r"(?P<directives>(?:^[ \t]*<!--[^\n]*-->[ \t]*\n)*)"
    r"^```(?P<lang>python|py)[ \t]*\n(?P<code>.*?)^```",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Snippet:
    path: Path
    line: int
    code: str
    directives: str

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"

    def has(self, directive: str) -> bool:
        return f"docs-test: {directive}" in self.directives


def _collect() -> list[Snippet]:
    snippets: list[Snippet] = []
    for path in DOC_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in FENCE_RE.finditer(text):
            line = text.count("\n", 0, m.start("code")) + 1
            snippets.append(
                Snippet(
                    path=path,
                    line=line,
                    code=m.group("code"),
                    directives=m.group("directives") or "",
                )
            )
    return snippets


SNIPPETS = _collect()
CHECKED = [s for s in SNIPPETS if not s.has("skip")]


def test_documentation_snippets_were_found():
    """Guard against the fence regex silently matching nothing."""
    assert len(SNIPPETS) > 40, f"only found {len(SNIPPETS)} snippets — is the regex broken?"


#: Compile flag that permits top-level ``await`` / ``async for`` / ``async with``,
#: the way ``python -m asyncio`` and IPython do. Docs snippets are written in that
#: style deliberately, so they are valid *as documentation* even though a plain
#: ``compile()`` would reject them.
_ALLOW_TOP_LEVEL_AWAIT = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


def _compile(snippet: Snippet):
    return compile(snippet.code, snippet.id, "exec", flags=_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize("snippet", CHECKED, ids=lambda s: s.id)
def test_snippet_compiles(snippet: Snippet):
    """Every documented snippet must be syntactically valid Python.

    Top-level ``await`` is allowed — docs snippets use it to keep examples short.
    """
    try:
        _compile(snippet)
    except SyntaxError as exc:
        pytest.fail(f"{snippet.id} is not valid Python: {exc}")


def _actants_symbols() -> dict[str, object]:
    import actants

    return {name: getattr(actants, name) for name in actants.__all__ if name != "__version__"}


@pytest.mark.parametrize("snippet", CHECKED, ids=lambda s: s.id)
def test_snippet_imports_resolve(snippet: Snippet):
    """Every `from actants... import X` in the docs must name a real symbol."""
    tree = ast.parse(snippet.code, snippet.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("actants"):
            continue
        try:
            module = __import__(node.module, fromlist=["_"])
        except ImportError as exc:  # optional extra not installed in this env
            pytest.skip(f"{node.module} unavailable: {exc}")
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{snippet.id} imports {alias.name!r} from {node.module!r}, "
                "which does not exist. The docs are out of date."
            )


def _public_callables() -> dict[str, object]:
    """Public actants classes/functions the docs are likely to call by name."""
    import actants

    out: dict[str, object] = {}
    for name in actants.__all__:
        if name == "__version__":
            continue
        try:
            out[name] = getattr(actants, name)
        except Exception:  # pragma: no cover - lazy import of a missing extra
            continue
    return out


@pytest.mark.parametrize("snippet", CHECKED, ids=lambda s: s.id)
def test_snippet_keyword_arguments_exist(snippet: Snippet):
    """Keyword args passed to public actants callables must exist in the signature.

    This is what catches `LLM(provider=...)`-style drift: a snippet that passes a
    kwarg the constructor no longer accepts fails here instead of in a user's
    terminal.
    """
    symbols = _public_callables()
    tree = ast.parse(snippet.code, snippet.id)

    # Comparison snippets (migration guides) import the *other* framework's classes
    # under names we also export — e.g. CrewAI's `Agent`. Only check names this
    # snippet actually imported from actants.
    from_actants: set[str] = set()
    shadowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            target_set = from_actants if node.module.startswith("actants") else shadowed
            for alias in node.names:
                target_set.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in shadowed and name not in from_actants:
            continue
        target = symbols.get(name)
        if target is None or not (inspect.isclass(target) or inspect.isfunction(target)):
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kw:
            continue
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat at the call site
                continue
            assert kw.arg in sig.parameters, (
                f"{snippet.id} calls {node.func.id}({kw.arg}=...), but "
                f"{node.func.id} accepts only {sorted(sig.parameters)}. "
                "The docs are out of date."
            )


OFFLINE_RUNNABLE = [s for s in CHECKED if s.has("run")]


@pytest.mark.parametrize("snippet", OFFLINE_RUNNABLE, ids=lambda s: s.id)
def test_snippet_executes(snippet: Snippet):
    """Snippets marked `<!-- docs-test: run -->` are executed for real.

    Only tag snippets that need no network, no API key, and no running server.
    """
    namespace: dict[str, object] = {"__name__": "__docs_snippet__"}
    code = _compile(snippet)
    if code.co_flags & inspect.CO_COROUTINE:
        import asyncio

        asyncio.run(eval(code, namespace))  # noqa: S307 — snippet is repo-controlled
    else:
        exec(code, namespace)  # noqa: S102 — snippet is repo-controlled
