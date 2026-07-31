"""Run one benchmark task N times inside a framework's venv and report timings.

Invoked by ``run_benchmarks.py`` as a subprocess, once per (framework, task).
Prints a single JSON object to stdout.

Each iteration records:

* ``wall`` — end-to-end time for the framework call.
* ``wire`` — summed proxy-measured time Ollama spent on that iteration's
  HTTP requests (identical model work across frameworks).
* ``overhead`` — ``wall - wire``, the framework's own cost.
* ``requests`` — how many HTTP round trips the framework made.

Object construction (building the client/agent) is inside the timed region
because that is what a user's request handler actually does; frameworks that
make construction expensive should pay for it.

Usage:
  python measure_latency.py <task_module> <task_name> <model> <runs> <proxy_url>
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
import urllib.request

PROMPTS = {
    "task_completion": "Reply with exactly: OK",
    "task_tool_agent": "What is the weather in Paris? Use the get_weather tool.",
    "task_structured": "Extract the person: Ada Lovelace is 36 years old and lives in London.",
}


def _proxy_records(proxy_url: str) -> list[dict]:
    with urllib.request.urlopen(f"{proxy_url}/__bench__/records", timeout=30) as response:
        return json.loads(response.read())


def _proxy_reset(proxy_url: str) -> None:
    with urllib.request.urlopen(f"{proxy_url}/__bench__/reset", timeout=30) as response:
        response.read()


async def main() -> None:
    task_module, task_name, model, runs_s, proxy_url = sys.argv[1:6]
    runs = int(runs_s)

    module = importlib.import_module(task_module)
    func = getattr(module, task_name)
    prompt = PROMPTS[task_name]

    # One untimed warm-up: pays for lazy imports, connection pool setup, and
    # any first-call caching inside the framework, so the reported samples
    # measure steady-state behaviour rather than one-off initialisation.
    warmup_error = None
    try:
        await func(model, prompt)
    except Exception as exc:  # noqa: BLE001
        warmup_error = f"{type(exc).__name__}: {exc}"

    samples: list[dict] = []
    errors: list[str] = []
    for _ in range(runs):
        _proxy_reset(proxy_url)
        start = time.perf_counter()
        try:
            result = await func(model, prompt)
            ok = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            result, ok = None, False
        wall = time.perf_counter() - start

        records = _proxy_records(proxy_url)
        wire = sum(r["wire_seconds"] for r in records)
        eval_count = sum(r.get("eval_count", 0) or 0 for r in records)
        samples.append(
            {
                "wall": wall,
                "wire": wire,
                "overhead": wall - wire,
                "requests": len(records),
                "eval_count": eval_count,
                "ok": ok,
                "output": str(result)[:200] if ok else None,
            }
        )

    json.dump(
        {
            "task_module": task_module,
            "task": task_name,
            "model": model,
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            "warmup_error": warmup_error,
            "errors": errors,
            "samples": samples,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    asyncio.run(main())
