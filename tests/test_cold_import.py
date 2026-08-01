"""CI gate: ``import actants`` must stay under 200ms.

This is a wedge — every framework ships with multi-second imports. We don't.
The gate exists so a careless eager-import in __init__.py fails CI immediately.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

COLD_IMPORT_BUDGET_MS = 200.0
HOT_INSTANTIATE_BUDGET_MS = 5.0


def _measure_subprocess_import_ms() -> float:
    """Spawn a fresh Python and time ``import actants`` end-to-end."""
    code = (
        "import time, sys; "
        "t = time.perf_counter(); "
        "import actants; "
        "print((time.perf_counter() - t) * 1000)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def test_cold_import_under_200ms():
    """Median of 3 cold imports must be under 200ms."""
    samples = sorted([_measure_subprocess_import_ms() for _ in range(3)])
    median = samples[1]
    assert median < COLD_IMPORT_BUDGET_MS, (
        f"Cold import too slow: {median:.1f} ms (budget: {COLD_IMPORT_BUDGET_MS}). "
        f"Samples: {samples}. Add eager imports to actants/__init__.py only "
        f"after profiling and confirming budget headroom."
    )


def test_no_attribute_access_keeps_import_minimal():
    """``import actants`` alone (no attribute access) must be near-instant."""
    samples = []
    for _ in range(3):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import time; t=time.perf_counter(); "
                "import actants; "
                "print((time.perf_counter()-t)*1000)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        samples.append(float(out.stdout.strip()))
    samples.sort()
    median = samples[1]
    # 50ms covers all reasonable platforms; a few stdlib imports + PEP 562.
    assert median < 50.0, f"Bare import too slow: {median:.1f} ms (samples: {samples})"


#: Submodules a user may legitimately import first, before anything else in actants.
#: Each one must stand on its own — no import-order-dependent success.
SUBMODULES = [
    "actants.cache",
    "actants.cache.memory",
    "actants.cache.protocol",
    "actants.cache.request",
    "actants.cache.semantic",
    "actants.agents.agent",
    "actants.llm.client",
    "actants.llm.base",
    "actants.tools.registry",
    "actants.cost.tracker",
    "actants.policies.retry",
    "actants.policies.fallback",
]


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports_first_without_a_cycle(module: str):
    """Importing any submodule *first* must not hit a partially-initialized module.

    ``actants.cache.protocol`` annotated with a runtime import of ``actants.llm.base``,
    which pulls in ``actants.llm.__init__`` -> ``actants.llm.client``, which imports
    ``actants.cache.protocol`` back. ``import actants.cache.semantic`` as the first
    actants import therefore raised ImportError, while the same import after
    ``import actants`` worked — so the whole test suite missed it.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`import {module}` as the first actants import failed:\n{result.stderr}"
    )


def test_agent_instantiation_after_warm_import_is_fast():
    """Once warm, instantiating an Agent should be sub-5ms."""
    from actants.agents import Agent
    from actants.llm.client import LLM
    from actants.testing import FakeLLMProvider

    provider = FakeLLMProvider()
    llm = LLM(provider=provider, model="fake")
    Agent(llm=llm)

    samples = []
    for _ in range(50):
        start = time.perf_counter()
        Agent(llm=llm)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    median = samples[len(samples) // 2]
    assert median < HOT_INSTANTIATE_BUDGET_MS, (
        f"Agent() instantiation too slow: median {median:.3f} ms "
        f"(budget {HOT_INSTANTIATE_BUDGET_MS} ms)"
    )
