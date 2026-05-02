"""Benchmark harness — measures import time and agent instantiation across
agentic-kit and competitor frameworks.

Run::

    python -m agentic_kit.bench
    python -m agentic_kit.bench --compare langchain,llamaindex,pydantic-ai
    python -m agentic_kit.bench --md > BENCHMARKS.md
"""

from __future__ import annotations

from agentic_kit.bench.runner import (
    BenchmarkResult,
    Framework,
    measure_import_ms,
    measure_instantiate_ms,
    run_all,
)

__all__ = [
    "BenchmarkResult",
    "Framework",
    "measure_import_ms",
    "measure_instantiate_ms",
    "run_all",
]
