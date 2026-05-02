"""Sanity tests for the benchmark harness."""

from __future__ import annotations

from agentic_kit.bench import (
    BenchmarkResult,
    Framework,
    measure_import_ms,
    measure_instantiate_ms,
    run_all,
)
from agentic_kit.bench.runner import (
    COMPETITORS,
    format_table,
    measure_bare_import_ms,
)


def test_competitors_list_includes_agentic_kit():
    names = {f.name for f in COMPETITORS}
    assert "agentic-kit" in names


def test_measure_bare_import_returns_samples_for_self():
    fw = next(f for f in COMPETITORS if f.name == "agentic-kit")
    samples = measure_bare_import_ms(fw, samples=2)
    assert len(samples) == 2
    assert all(s > 0 for s in samples)


def test_measure_import_returns_samples_for_self():
    fw = next(f for f in COMPETITORS if f.name == "agentic-kit")
    samples = measure_import_ms(fw, samples=2)
    assert len(samples) == 2


def test_measure_instantiate_returns_samples_for_self():
    fw = next(f for f in COMPETITORS if f.name == "agentic-kit")
    samples = measure_instantiate_ms(fw, samples=2)
    assert len(samples) == 2


def test_run_all_skips_uninstalled():
    fake = Framework(
        name="definitely-not-real",
        package="this_package_does_not_exist_xyzzy",
        bare_import_code="import this_package_does_not_exist_xyzzy",
        import_code="import this_package_does_not_exist_xyzzy",
        instantiate_code=None,
    )
    results = run_all([fake], skip_uninstalled=True)
    assert len(results) == 1
    assert results[0].error == "not installed"


def test_format_table_renders_markdown():
    fw = next(f for f in COMPETITORS if f.name == "agentic-kit")
    results = [BenchmarkResult(framework=fw.name, bare_import_samples_ms=[1.5, 1.6, 1.4])]
    table = format_table(results)
    assert "| Framework |" in table
    assert "agentic-kit" in table


def test_agentic_kit_bare_import_is_under_50ms():
    """Hard line in the sand: our PEP 562 wedge must hold."""
    fw = next(f for f in COMPETITORS if f.name == "agentic-kit")
    samples = measure_bare_import_ms(fw, samples=3)
    median = sorted(samples)[1]
    assert median < 50.0, (
        f"bare import too slow: {median:.1f}ms (samples: {samples}). "
        "Someone added an eager top-level import — check __init__.py."
    )
