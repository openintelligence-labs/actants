"""Subprocess-based benchmark runner.

Each measurement spawns a fresh Python process so we measure cold-start cost
honestly — no warm caches, no module-cache pollution from prior runs.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Framework:
    """A competitor framework to benchmark.

    Two import variants matter:
      * ``bare_import_code`` — ``import <pkg>`` only (PEP 562 lazy frameworks win)
      * ``import_code`` — typical first usable import a real app would write

    ``instantiate_code`` builds the simplest valid agent. Set any to ``None``
    to skip that measurement.
    """

    name: str
    package: str
    bare_import_code: str | None
    import_code: str | None
    instantiate_code: str | None


# Best-effort competitor profiles. ``instantiate_code`` aims for the simplest
# valid "agent with one tool" pattern per each framework's docs.
COMPETITORS: list[Framework] = [
    Framework(
        name="actants",
        package="actants",
        bare_import_code="import actants",
        import_code="from actants import Agent, LLM",
        instantiate_code=(
            "from actants.testing import FakeLLMProvider; "
            "from actants import Agent, LLM; "
            "Agent(llm=LLM(provider=FakeLLMProvider(), model='fake'))"
        ),
    ),
    Framework(
        name="langchain",
        package="langchain_core",
        bare_import_code="import langchain_core",
        import_code="from langchain_core.prompts import PromptTemplate",
        instantiate_code=None,  # requires API key to instantiate cleanly
    ),
    Framework(
        name="langgraph",
        package="langgraph",
        bare_import_code="import langgraph",
        import_code="from langgraph.graph import StateGraph",
        instantiate_code=None,
    ),
    Framework(
        name="llama_index",
        package="llama_index.core",
        bare_import_code="import llama_index.core",
        import_code="from llama_index.core.agent.workflow import FunctionAgent",
        instantiate_code=None,
    ),
    Framework(
        name="pydantic_ai",
        package="pydantic_ai",
        bare_import_code="import pydantic_ai",
        import_code="from pydantic_ai import Agent",
        instantiate_code=(
            "from pydantic_ai import Agent; "
            "from pydantic_ai.models.test import TestModel; "
            "Agent(TestModel())"
        ),
    ),
    Framework(
        name="crewai",
        package="crewai",
        bare_import_code="import crewai",
        import_code="from crewai import Agent",
        instantiate_code=None,
    ),
    Framework(
        name="smolagents",
        package="smolagents",
        bare_import_code="import smolagents",
        import_code="from smolagents import CodeAgent",
        instantiate_code=None,
    ),
    Framework(
        name="agno",
        package="agno",
        bare_import_code="import agno",
        import_code="from agno.agent import Agent",
        instantiate_code=None,
    ),
    Framework(
        name="autogen_agentchat",
        package="autogen_agentchat",
        bare_import_code="import autogen_agentchat",
        import_code="from autogen_agentchat.agents import AssistantAgent",
        instantiate_code=None,
    ),
]


@dataclass
class BenchmarkResult:
    framework: str
    bare_import_samples_ms: list[float] = field(default_factory=list)
    import_samples_ms: list[float] = field(default_factory=list)
    instantiate_samples_ms: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def bare_import_median_ms(self) -> float | None:
        return (
            statistics.median(self.bare_import_samples_ms) if self.bare_import_samples_ms else None
        )

    @property
    def import_median_ms(self) -> float | None:
        return statistics.median(self.import_samples_ms) if self.import_samples_ms else None

    @property
    def instantiate_median_ms(self) -> float | None:
        if not self.instantiate_samples_ms:
            return None
        return statistics.median(self.instantiate_samples_ms)


def _run_subprocess(code: str) -> float:
    """Run ``code`` in a fresh Python and return wall-clock ms it took."""
    wrapper = (
        "import time\n"
        "_start = time.perf_counter()\n"
        f"{code}\n"
        "print((time.perf_counter() - _start) * 1000)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def measure_bare_import_ms(framework: Framework, *, samples: int = 3) -> list[float]:
    if framework.bare_import_code is None:
        return []
    return sorted(_run_subprocess(framework.bare_import_code) for _ in range(samples))


def measure_import_ms(framework: Framework, *, samples: int = 3) -> list[float]:
    if framework.import_code is None:
        return []
    return sorted(_run_subprocess(framework.import_code) for _ in range(samples))


def measure_instantiate_ms(framework: Framework, *, samples: int = 3) -> list[float]:
    if framework.instantiate_code is None:
        return []
    return sorted(_run_subprocess(framework.instantiate_code) for _ in range(samples))


def _is_installed(framework: Framework) -> bool:
    try:
        __import__(framework.package)
        return True
    except ImportError:
        return False


def run_all(
    frameworks: list[Framework] | None = None,
    *,
    samples: int = 3,
    skip_uninstalled: bool = True,
) -> list[BenchmarkResult]:
    targets = frameworks if frameworks is not None else COMPETITORS
    results: list[BenchmarkResult] = []
    for fw in targets:
        result = BenchmarkResult(framework=fw.name)
        if skip_uninstalled and not _is_installed(fw):
            result.error = "not installed"
            results.append(result)
            continue
        try:
            result.bare_import_samples_ms = measure_bare_import_ms(fw, samples=samples)
            result.import_samples_ms = measure_import_ms(fw, samples=samples)
            result.instantiate_samples_ms = measure_instantiate_ms(fw, samples=samples)
        except subprocess.CalledProcessError as exc:
            result.error = (exc.stderr or str(exc))[:200]
        results.append(result)
    return results


def format_table(results: list[BenchmarkResult]) -> str:
    """Format as a Markdown table — what we publish in the README."""
    lines = [
        "| Framework | Bare `import` (ms) | First-use import (ms) | Instantiate (ms) |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.framework} | _{r.error}_ | — | — |")
            continue
        bare = f"{r.bare_import_median_ms:.1f}" if r.bare_import_median_ms is not None else "—"
        imp = f"{r.import_median_ms:.1f}" if r.import_median_ms is not None else "—"
        inst = f"{r.instantiate_median_ms:.2f}" if r.instantiate_median_ms is not None else "—"
        lines.append(f"| {r.framework} | {bare} | {imp} | {inst} |")
    return "\n".join(lines)
