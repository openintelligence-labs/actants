"""Benchmark harness — measures import time and agent instantiation across
actants and competitor frameworks.

Run::

    python -m actants.bench
    python -m actants.bench --compare langchain,llamaindex,pydantic-ai
    python -m actants.bench --md > BENCHMARKS.md
"""

from __future__ import annotations

from actants.bench.runner import (
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
