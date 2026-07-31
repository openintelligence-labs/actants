"""Benchmark runner: one command produces every number in docs/BENCHMARK.md.

    python benchmarks/run_benchmarks.py --runs 7

What it does, in order:

1. Creates (or reuses) one isolated venv per framework under ``--venv-dir``,
   installing the pinned versions from ``requirements/``. Nothing is installed
   into the actants development venv.
2. Records the resolved version of every package in every venv.
3. Measures install footprint: package count and site-packages size.
4. Measures cold import time and post-import RSS, best-of-N cold subprocesses.
5. Warms the model, then measures the three tasks for every framework,
   rotating framework order between rounds so that any thermal or cache drift
   is spread across frameworks rather than concentrated on whichever one runs
   first.
6. Counts LOC and imports from the task files.
7. Writes ``benchmarks/results.json``.

All latency numbers are reported as p50/p95 over the samples, never as means,
because the distribution is skewed by occasional model-side stalls.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
REQ_DIR = HERE / "requirements"

# framework key -> (requirements file stem, task module, import statement to time)
#
# The import statement is what the benchmark tasks actually import, not the
# bare package name. `actants` and `langchain` are both lazy packages: their
# top-level `import` is a sub-millisecond stub that loads nothing, so timing
# it would measure the stub rather than the framework. Charging each
# framework for the symbols its tasks use is the honest comparison.
FRAMEWORKS: dict[str, tuple[str, str, str]] = {
    "actants": (
        "actants",
        "task_actants",
        "from actants import LLM, LLMSettings, ToolRegistry",
    ),
    "langchain": (
        "langchain",
        "task_langchain",
        "from langchain_ollama import ChatOllama\n"
        "from langchain.agents import create_agent\n"
        "from langchain_core.tools import tool",
    ),
    "pydantic_ai": (
        "pydantic_ai",
        "task_pydantic_ai",
        "from pydantic_ai import Agent, NativeOutput\n"
        "from pydantic_ai.models.openai import OpenAIChatModel\n"
        "from pydantic_ai.providers.ollama import OllamaProvider",
    ),
    "llama_index": (
        "llama_index",
        "task_llama_index",
        "from llama_index.llms.ollama import Ollama\n"
        "from llama_index.core.agent.workflow import FunctionAgent\n"
        "from llama_index.core.tools import FunctionTool",
    ),
    "raw": ("raw", "task_raw", "from ollama import AsyncClient"),
}

TASKS = ("task_completion", "task_tool_agent", "task_structured")

PROXY_PORT = 11500
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def machine_specs() -> dict:
    specs = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    if sys.platform == "darwin":
        for key, cmd in (
            ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
        ):
            with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
                specs[key] = subprocess.run(
                    cmd, capture_output=True, text=True, check=True
                ).stdout.strip()
    return specs


def ollama_version() -> str:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=10) as response:
            return json.loads(response.read()).get("version", "unknown")
    except (urllib.error.URLError, OSError):
        return "unknown"


def warm_model(model: str, keep_alive: str = "60m") -> None:
    """Load the model into VRAM before any timing, and pin it there.

    Without this, whichever framework happens to run first absorbs the entire
    multi-second model load and looks catastrophically slow.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": keep_alive,
        }
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        response.read()


# --------------------------------------------------------------------------
# venvs
# --------------------------------------------------------------------------
def _uv() -> str | None:
    return shutil.which("uv")


def ensure_venvs(venv_dir: Path, force: bool) -> dict[str, Path]:
    venv_dir.mkdir(parents=True, exist_ok=True)
    pythons: dict[str, Path] = {}
    uv = _uv()

    for name, (req_stem, _, _) in FRAMEWORKS.items():
        target = venv_dir / name
        python = target / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        req = REQ_DIR / f"{req_stem}.txt"

        if force and target.exists():
            shutil.rmtree(target)

        if not python.exists():
            print(f"  creating venv: {name}", flush=True)
            if uv:
                subprocess.run(
                    [uv, "venv", "--python", "3.13", str(target)], check=True, capture_output=True
                )
            else:
                subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)

            print(f"  installing:    {name}", flush=True)
            if uv:
                subprocess.run(
                    [uv, "pip", "install", "--python", str(python), "-q", "-r", str(req)],
                    check=True,
                )
            else:
                subprocess.run(
                    [str(python), "-m", "pip", "install", "-q", "-r", str(req)], check=True
                )
        pythons[name] = python
    return pythons


def freeze(python: Path) -> dict[str, str]:
    uv = _uv()
    cmd = (
        [uv, "pip", "freeze", "--python", str(python)]
        if uv
        else [str(python), "-m", "pip", "freeze"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    versions: dict[str, str] = {}
    for line in out.splitlines():
        if "==" in line:
            pkg, _, version = line.partition("==")
            versions[pkg.strip().lower()] = version.strip()
    return versions


def site_packages_bytes(python: Path) -> int:
    out = subprocess.run(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    root = Path(out)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def measure_imports(pythons: dict[str, Path], runs: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, (_, _, statement) in FRAMEWORKS.items():
        samples, rss = [], []
        for _ in range(runs):
            proc = subprocess.run(
                [str(pythons[name]), str(HERE / "measure_import.py"), statement],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(proc.stdout)
            samples.append(data["import_seconds"])
            rss.append(data["rss_bytes"])
        results[name] = {
            "statement": statement,
            "import_seconds_best": min(samples),
            "import_seconds_p50": statistics.median(samples),
            "rss_bytes_median": statistics.median(rss),
            "modules_loaded": data["modules_loaded"],
            "samples": samples,
        }
        print(
            f"  {name:<12} import best={min(samples) * 1000:7.1f}ms  "
            f"rss={statistics.median(rss) / 1e6:6.1f}MB",
            flush=True,
        )
    return results


def measure_latency(
    pythons: dict[str, Path], model: str, runs: int, rounds: int, seed: int
) -> dict[str, dict]:
    """Run every (framework, task) pair, rotating order between rounds."""
    raw: dict[str, dict[str, list[dict]]] = {n: {t: [] for t in TASKS} for n in FRAMEWORKS}
    errors: dict[str, list[str]] = {n: [] for n in FRAMEWORKS}
    rng = random.Random(seed)

    per_round = max(1, runs // rounds)
    for round_index in range(rounds):
        order = list(FRAMEWORKS)
        rng.shuffle(order)
        print(f"  round {round_index + 1}/{rounds}: {' -> '.join(order)}", flush=True)
        for name in order:
            _, task_module, _ = FRAMEWORKS[name]
            for task in TASKS:
                proc = subprocess.run(
                    [
                        str(pythons[name]),
                        str(HERE / "measure_latency.py"),
                        task_module,
                        task,
                        model,
                        str(per_round),
                        PROXY_URL,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(TASKS_DIR),
                    env=_task_env(),
                )
                if proc.returncode != 0:
                    errors[name].append(f"{task}: exit {proc.returncode}: {proc.stderr[-400:]}")
                    continue
                data = json.loads(proc.stdout)
                if data.get("warmup_error"):
                    errors[name].append(f"{task} warmup: {data['warmup_error']}")
                errors[name].extend(f"{task}: {e}" for e in data.get("errors", []))

                good = [s for s in data["samples"] if s["ok"]]
                # A sample with zero proxied requests means the framework
                # bypassed the proxy, so `wire` is 0 and `overhead` would be
                # the whole model time -- a wildly flattering artefact. Drop
                # it loudly rather than let it into the results.
                bypassed = [s for s in good if s["requests"] == 0]
                if bypassed:
                    errors[name].append(
                        f"{task}: {len(bypassed)} sample(s) bypassed the proxy and were discarded"
                    )
                raw[name][task].extend(s for s in good if s["requests"] > 0)

    return {"raw": raw, "errors": errors}


def _task_env() -> dict[str, str]:
    """Point every framework at the recording proxy rather than Ollama.

    Task files read ``BENCH_OLLAMA_URL`` and pass the host explicitly to their
    client constructor. Relying on each framework's own env-var support was
    tried first and silently failed for three of the five -- they ignored the
    variable, went straight to 11434, and reported the full model time as
    "framework overhead". The runner now asserts that every task actually
    routed through the proxy (see ``requests`` in the samples).
    """
    import os

    env = dict(os.environ)
    env.update({"BENCH_OLLAMA_URL": PROXY_URL, "PYTHONPATH": str(TASKS_DIR)})
    return env


def summarise(samples: list[dict]) -> dict | None:
    if not samples:
        return None

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[index]

    wall = [s["wall"] for s in samples]
    overhead = [s["overhead"] for s in samples]
    wire = [s["wire"] for s in samples]
    return {
        "n": len(samples),
        "wall_p50": pct(wall, 0.50),
        "wall_p95": pct(wall, 0.95),
        "overhead_p50": pct(overhead, 0.50),
        "overhead_p95": pct(overhead, 0.95),
        "wire_p50": pct(wire, 0.50),
        "requests_median": statistics.median([s["requests"] for s in samples]),
        "eval_count_median": statistics.median([s["eval_count"] for s in samples]),
    }


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--runs", type=int, default=7, help="latency samples per framework/task")
    parser.add_argument("--rounds", type=int, default=7, help="interleaved rounds (order rotates)")
    parser.add_argument("--import-runs", type=int, default=5)
    parser.add_argument("--venv-dir", type=Path, default=Path("/tmp/actants-bench-venvs"))
    parser.add_argument("--force-venvs", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    print("actants benchmark suite")
    print("=" * 60)

    print("\n[1/6] environment")
    specs = machine_specs()
    version = ollama_version()
    print(f"  {specs.get('cpu', specs['processor'])} | ollama {version} | model {args.model}")

    print("\n[2/6] venvs")
    pythons = ensure_venvs(args.venv_dir, args.force_venvs)
    versions = {name: freeze(python) for name, python in pythons.items()}

    print("\n[3/6] install footprint")
    footprint = {}
    for name, python in pythons.items():
        size = site_packages_bytes(python)
        footprint[name] = {"packages": len(versions[name]), "site_packages_bytes": size}
        print(f"  {name:<12} {len(versions[name]):3d} packages  {size / 1e6:7.1f} MB", flush=True)

    print("\n[4/6] cold import")
    imports = measure_imports(pythons, args.import_runs)

    print("\n[5/6] latency (proxy on, model warmed)")
    sys.path.insert(0, str(HERE))
    from ollama_proxy import serve  # noqa: PLC0415

    serve(PROXY_PORT)
    time.sleep(0.5)
    print(f"  warming {args.model} ...", flush=True)
    warm_model(args.model)

    latency = measure_latency(pythons, args.model, args.runs, args.rounds, args.seed)

    summary: dict[str, dict] = {}
    for name in FRAMEWORKS:
        summary[name] = {t: summarise(latency["raw"][name][t]) for t in TASKS}

    print("\n[6/6] lines of code")
    loc = json.loads(
        subprocess.run(
            [sys.executable, str(HERE / "measure_loc.py"), str(TASKS_DIR)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "ollama_version": version,
        "machine": specs,
        "config": {
            "runs": args.runs,
            "rounds": args.rounds,
            "import_runs": args.import_runs,
            "seed": args.seed,
        },
        "versions": versions,
        "footprint": footprint,
        "imports": imports,
        "latency": summary,
        "latency_errors": {k: v for k, v in latency["errors"].items() if v},
        "loc": loc,
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nwrote {args.out}")
    _print_tables(results)


def _print_tables(results: dict) -> None:
    print("\n" + "=" * 60)
    print("LATENCY (p50 wall / p50 framework overhead), seconds")
    print("=" * 60)
    for task in TASKS:
        print(f"\n{task}")
        rows = []
        for name in FRAMEWORKS:
            stats = results["latency"][name].get(task)
            if stats:
                rows.append((stats["overhead_p50"], name, stats))
        for _, name, stats in sorted(rows):
            print(
                f"  {name:<12} wall p50={stats['wall_p50']:6.3f}s  "
                f"overhead p50={stats['overhead_p50'] * 1000:7.2f}ms  "
                f"p95={stats['overhead_p95'] * 1000:7.2f}ms  "
                f"reqs={stats['requests_median']:.0f}  n={stats['n']}"
            )
    if results["latency_errors"]:
        print("\nERRORS")
        for name, errs in results["latency_errors"].items():
            for err in errs[:3]:
                print(f"  {name}: {err[:160]}")


if __name__ == "__main__":
    main()
