# benchmarks

Reproducible comparison of `actants` against LangChain, Pydantic AI,
LlamaIndex, and the raw `ollama` client. Results and full methodology live in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

## Run it

Requires a local [Ollama](https://ollama.com) and one model:

```bash
ollama pull qwen2.5:7b
python benchmarks/run_benchmarks.py --runs 7
```

The runner builds one throwaway venv per framework under `--venv-dir`
(default `/tmp/actants-bench-venvs`) from the pinned lockfiles in
`requirements/`. It never installs into the actants development venv. Writes
`benchmarks/results.json`.

Useful flags: `--model`, `--runs`, `--rounds`, `--import-runs`,
`--force-venvs`, `--seed`.

## Layout

| Path | What it does |
|---|---|
| `run_benchmarks.py` | Orchestrator — builds venvs, runs every measurement, writes `results.json` |
| `ollama_proxy.py` | Recording proxy that separates model time from framework overhead |
| `measure_import.py` | Cold import time + post-import RSS, one cold interpreter per sample |
| `measure_latency.py` | Runs one task N times inside a framework's venv |
| `measure_loc.py` | Counts LOC and imports from the `# LOC_x_START` blocks |
| `tasks/task_*.py` | The three tasks implemented per framework — these are the snippets in the doc |
| `requirements/*.txt` | Pinned lockfile per comparison venv |

## Why a proxy

Model time (hundreds of ms) dwarfs framework time (single-digit ms), so
end-to-end wall clock cannot distinguish frameworks. Every framework is
pointed at `ollama_proxy.py`, which forwards to Ollama and records how long
each upstream request took. `overhead = wall - wire` isolates the framework's
own cost from identical model work.

Tasks read the proxy URL from `BENCH_OLLAMA_URL` and pass it explicitly to
their client constructor. Relying on each framework's own env-var support was
tried first and silently failed for three of five — they went straight to
`localhost:11434` and reported all of model time as framework overhead. The
runner now discards any sample that made zero proxied requests.

## Editing tasks

Imports live *inside* the `# LOC_A_START` / `# LOC_A_END` markers on purpose,
so the counts in the doc are measured from the code that actually runs.
`pyproject.toml` carries a scoped `E402` exemption for `tasks/task_*.py`.
Keep the three tasks semantically identical across frameworks — same prompt,
same tool, same output model — or the comparison stops meaning anything.
