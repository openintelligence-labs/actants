# actants benchmark

A reproducible comparison of `actants` against LangChain, Pydantic AI,
LlamaIndex, and the raw `ollama` client, on the things that actually differ
between SDKs: what you install, what you import, what you write, and what the
framework costs you on top of the model.

Every number below was produced by `benchmarks/run_benchmarks.py` on one
machine, in one sitting, against one local model. The raw output is committed
as `benchmarks/results.json`. Reproduction commands are at the bottom.

**Summary in one line:** `actants` has the smallest install and the fastest
import of the four frameworks tested, and its per-call overhead is within
noise of writing raw HTTP by hand. It loses on tool ergonomics — registering
a tool takes twice the code of LangChain's `@tool`, because released 0.5.3
does not infer schemas from type hints.

---

## What is being measured

Model time dominates everything. A single completion against a 7B model on
this machine takes ~150ms; the frameworks differ by ~5-10ms. If you only
measure end-to-end wall time you will conclude that all five are identical,
which is true but useless.

So every framework is pointed at a recording proxy
(`benchmarks/ollama_proxy.py`) that forwards to Ollama and times each upstream
request. That gives two numbers per run:

- **wire** — time Ollama spent. Identical work across frameworks.
- **overhead** = `wall - wire` — the framework's own cost: building the
  request, serialising the schema, parsing the response, running the agent
  loop.

**Overhead is the number that distinguishes these libraries.** Wall time is
reported too, so you can see how small the differences are in context.

### Honesty notes about the method

- All latency numbers are p50/p95 over 7 samples, never means. The
  distribution is skewed by occasional model-side stalls.
- The model is warmed and pinned in VRAM (`keep_alive`) before any timing.
  Without this, whichever framework ran first would absorb the multi-second
  model load.
- Framework order is shuffled between each of the 7 rounds (seeded, so it
  reproduces), so thermal drift and cache warming are spread across all five
  rather than concentrated on the one that happens to go first.
- Each task constructs its client inside the timed region, because that is
  what a request handler does. This penalises frameworks with expensive
  construction — see the LlamaIndex note, which is a real cost but one you
  can avoid by hoisting the object.
- **A trap worth documenting:** the first attempt routed frameworks to the
  proxy via `OLLAMA_HOST` / `OLLAMA_BASE_URL`. Three of the five silently
  ignored it, connected straight to port 11434, and therefore recorded
  `wire = 0` — reporting the entire model time as "framework overhead". The
  tasks now pass the host explicitly to each client constructor, and the
  runner discards any sample that made zero proxied requests. If you fork
  this benchmark, keep that guard.

---

## Machine and versions

Single-machine numbers. Do not treat small deltas as portable.

| | |
|---|---|
| CPU | Apple M4 Pro |
| Memory | 48 GiB |
| OS | macOS 15.7.3 (arm64) |
| Python | 3.13.5 |
| Ollama | 0.32.4 |
| Model | `qwen2.5:7b` (Q4_K_M, 7.6B) |
| Date | 2026-07-31 |
| Config | 7 latency samples, 7 rounds, 5 import samples, seed 20260730 |

Frameworks under test, as resolved into isolated venvs
(full lockfiles in `benchmarks/requirements/`):

| Framework | Version | Direct install |
|---|---|---|
| actants | 0.5.3 | `actants` |
| LangChain | langchain 1.3.14, langchain-core 1.5.3, langchain-ollama 1.1.0, langgraph 1.2.10 | `langchain langchain-ollama` |
| Pydantic AI | pydantic-ai 2.21.0 | `pydantic-ai` |
| LlamaIndex | llama-index-core 0.14.23, llama-index-llms-ollama 0.10.1 | `llama-index-core llama-index-llms-ollama` |
| raw | ollama 0.6.2, httpx 0.28.1 | `ollama httpx` |

All four frameworks installed cleanly. **Nothing was skipped.** All five share
identical `httpx` 0.28.1 and `pydantic` 2.13.4, so those are not a source of
difference.

`raw` is the control: the `ollama` Python client with a hand-written tool
loop. It is not a framework and it is not a fair competitor on features — it
exists to show what the abstractions cost.

---

## 1. Install footprint

Minimal "call an LLM + one tool" setup, in a fresh venv.

| Framework | Packages | site-packages |
|---|---|---|
| raw | 12 | 11.3 MB |
| **actants** | **18** | **14.1 MB** |
| LangChain | 38 | 35.9 MB |
| Pydantic AI | 98 | 105.7 MB |
| LlamaIndex | 63 | 126.7 MB |

`actants` pulls 6 packages beyond the raw floor and is the smallest of the
four frameworks — 2.5x smaller than LangChain, 7.5x smaller than Pydantic AI
by package count.

The gap is mostly about what ships by default. Pydantic AI's base install
includes the OpenAI, Anthropic, and Google clients, MCP, and Logfire.
LlamaIndex pulls `numpy`, `nltk`, `pillow`, `sqlalchemy`, and `tiktoken` into
`llama-index-core`. `actants` puts every provider behind an extra, so the
default install is Ollama-only. That is a deliberate trade, not a free win:
if you want OpenAI, you install the extra and the gap narrows.

---

## 2. Cold import

Wall time to import the symbols the benchmark tasks actually use, in a cold
subprocess, best-of-5. RSS is process peak after import.

| Framework | Best | p50 | RSS | sys.modules |
|---|---|---|---|---|
| **actants** | **96.9 ms** | 99.1 ms | 42.6 MB | 396 |
| raw | 116.8 ms | 123.8 ms | 45.7 MB | 356 |
| LangChain | 343.2 ms | 348.9 ms | 82.6 MB | 1058 |
| LlamaIndex | 565.7 ms | 616.5 ms | 116.0 MB | 1441 |
| Pydantic AI | 742.5 ms | 830.0 ms | 123.9 MB | 2493 |

`actants` imports 3.5x faster than LangChain and 7.7x faster than Pydantic AI.

It also comes out ~20 ms ahead of the raw `ollama` client, which is worth
flagging as *suspicious rather than impressive*: `actants` loads 396 modules
to raw's 356, and both end up importing `httpx` and `pydantic`. A library
that loads strictly more modules should not import faster. The most likely
explanation is filesystem cache and import-order effects between separate
venvs, not a genuine advantage. Treat `actants` and raw as tied on import,
and treat the 3.5x-7.7x gaps against the larger frameworks — which track
module counts of 1058-2493 — as the real signal.

**This metric was initially measured wrong, and the correction matters.**
Timing a bare `import actants` gives 1.5 ms, and `import langchain` gives
0.3 ms — both are lazy stubs that load nothing. Comparing those two numbers
would have been a meaningless win. The table above charges each framework for
importing what its tasks actually use (e.g. `from langchain_ollama import
ChatOllama` + `create_agent` + `tool`). That is the cost a real program pays.

Import time matters for CLIs and serverless cold starts. For a long-running
server it is a one-off, and you should weight it accordingly.

---

## 3. Latency

Three tasks, 7 samples each, order shuffled across rounds. `overhead` is the
framework's own cost; `wall` is end-to-end including the model.

**Model time dominates.** The wall p50s below are nearly identical across
frameworks because every framework is doing the same model work. The overhead
column is the real comparison, and it is small in absolute terms for everyone
except LlamaIndex's agent.

### (a) One completion

| Framework | Overhead p50 | Overhead p95 | Wall p50 | HTTP reqs |
|---|---|---|---|---|
| raw | 5.58 ms | 14.53 ms | 0.152 s | 1 |
| **actants** | **6.03 ms** | **7.74 ms** | 0.145 s | 1 |
| Pydantic AI | 10.37 ms | 13.07 ms | 0.153 s | 1 |
| LangChain | 10.40 ms | 11.34 ms | 0.152 s | 1 |
| LlamaIndex | 11.80 ms | 33.19 ms | 0.156 s | 2 |

`actants` is statistically tied with raw at p50 (0.45 ms apart, well inside
run-to-run noise) and has the tightest p95 of all five. The three other
frameworks cost roughly 4-6 ms more per call than raw.

### (b) Agent with one tool (two model turns)

| Framework | Overhead p50 | Overhead p95 | Wall p50 | HTTP reqs |
|---|---|---|---|---|
| **actants** | **7.58 ms** | 16.68 ms | 1.160 s | 2 |
| raw | 7.62 ms | 11.82 ms | 1.171 s | 2 |
| Pydantic AI | 12.88 ms | 19.15 ms | 1.082 s | 2 |
| LangChain | 17.33 ms | 45.43 ms | 1.267 s | 2 |
| LlamaIndex | 951.32 ms | 997.39 ms | 2.095 s | 3 |

`actants` and raw are tied (0.04 ms apart — noise). Per tool-calling turn
that is ~3.8 ms of framework overhead for actants versus ~8.7 ms for LangChain.

**The LlamaIndex number is real and reproducible, not an artefact.** Measured
independently with `time.process_time()`: one tool run costs 2.12 s wall and
**0.95 s of actual CPU** inside the process. Five consecutive runs landed
between 907 and 976 ms. The cost is `FunctionAgent`'s workflow engine — the
event bus and step dispatch in `llama-index-workflows` — not model time, not
the network, and not a sleep. On a 7B local model that nearly doubles the
latency of a two-turn tool call.

### (c) Structured output into a pydantic model

| Framework | Overhead p50 | Overhead p95 | Wall p50 | HTTP reqs |
|---|---|---|---|---|
| raw | 6.21 ms | 8.17 ms | 0.758 s | 1 |
| **actants** | **6.21 ms** | 8.03 ms | 0.753 s | 1 |
| Pydantic AI | 10.46 ms | 21.71 ms | 0.774 s | 1 |
| LangChain | 12.03 ms | 14.13 ms | 0.757 s | 1 |
| LlamaIndex | 12.40 ms | 22.46 ms | 0.755 s | 2 |

Identical p50 to raw, to the hundredth of a millisecond.

### Why LlamaIndex issues an extra request

Every LlamaIndex task shows one more HTTP request than the others. `Ollama`
defaults to `context_window=-1`, and `get_context_window()` then calls
`/api/show` to read the model's real context length. It is cached on the
instance, so it is **once per `Ollama` object, not once per request** — the
benchmark constructs a client per call, so it pays every time. A long-lived
module-level client pays it once, and passing `context_window=32768`
explicitly skips it entirely. The ~5 ms it costs is not the reason for the
agent result above.

---

## 4. Lines of code and concepts

Counted mechanically by `benchmarks/measure_loc.py` from the code that
actually runs (`benchmarks/tasks/task_*.py`). Blank and comment lines
excluded; imports counted as both LOC and imports; shared helpers charged to
every task that calls them.

| Framework | (a) completion | (b) tool agent | (c) structured | Total |
|---|---|---|---|---|
| LangChain | 5 (1 import) | **10** (2) | 8 (1) | **23** |
| LlamaIndex | 5 (1) | 11 (2) | 9 (2) | 25 |
| Pydantic AI | 10 (3) | 10 (0) | 12 (2) | 32 |
| **actants** | 5 (1) | **20** (1) | 8 (1) | **33** |
| raw | 5 (1) | 32 (1) | 12 (1) | 49 |

**`actants` loses this metric, and the reason is specific and fixable.**

For a single completion and for structured output, `actants` ties LangChain
and LlamaIndex at the shortest implementation. The entire deficit is task
(b): registering one tool takes 20 lines versus LangChain's 10.

The cause is that released 0.5.3 requires a hand-written JSON Schema:

```python
tools.register_function(
    "get_weather",
    "Get the current weather for a city.",
    get_weather,
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
```

versus LangChain:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and raining in {city}"
```

`input_schema` is nominally optional in 0.5.3, but omitting it silently
produces `{"type": "object", "properties": {}}` — the model never learns the
tool takes a `city`, and the tool call fails at runtime. **Passing the schema
explicitly is mandatory in 0.5.3, so the 20-line count is the honest one for
the released package.**

Schema inference from type annotations has since landed on `main` but is
**not yet released**: on current `main`,
`register_function("get_weather", "...", get_weather)` correctly infers
`{"city": {"type": "string"}}` and required-ness, which would bring task (b)
down to roughly LangChain's line count. The table above deliberately measures
the version you get from `pip install actants` today. It will be regenerated
when the inference ships in a release.

Two smaller notes against `actants` in this table: its completion and
structured tasks import `LLMSettings` alongside `LLM`, which a normal user
would not need — the benchmark uses it to point the client at the recording
proxy. Without that constraint `actants` would be one import lighter on two
of the three tasks. Conversely, Pydantic AI's 10-line completion is inflated
by three imports and a `_model()` helper needed because its Ollama provider
refuses to default to `localhost:11434` (see below).

Side-by-side snippets for all three tasks are in
[`benchmarks/tasks/`](../benchmarks/tasks/) — read them rather than trusting
the counts.

---

## 5. Interoperability findings

Things that cost real debugging time and are worth knowing before you choose.

**Pydantic AI's Ollama provider will not default to localhost.** `Agent("ollama:qwen2.5:7b")`
raises `UserError: Set the OLLAMA_BASE_URL environment variable or pass it via
OllamaProvider(base_url=...)`. Every other framework here defaults to
`http://localhost:11434`. Minor, but it is the first thing you hit.

**Pydantic AI's default structured-output mode fails on this model.**
`Agent(model, output_type=Person)` uses tool-call output mode, which raises
`UnexpectedModelBehavior: Exceeded maximum output retries (1)` against
`qwen2.5:7b` — the model does not reliably emit the synthetic `final_result`
tool call. `NativeOutput(Person)` selects JSON-schema-constrained decoding,
which is what the other four frameworks use by default, and works
immediately. The benchmark uses `NativeOutput` so all five are doing the same
thing. If you use Pydantic AI with small local models, know that the default
is the fragile path.

**`actants` 0.5.3 silently accepts a tool with no schema.** Covered above.
The failure is at model-call time, not registration time, which makes it
harder to catch than an exception would be. Fixed on `main` (annotations are
now read automatically), unreleased at the time of measurement.

---

## Where actants wins, ties, and loses

**Wins**

- Smallest install of the four frameworks: 18 packages / 14.1 MB, vs
  LangChain's 38 / 35.9 MB and Pydantic AI's 98 / 105.7 MB.
- Fastest cold import among the frameworks: 96.9 ms, 3.5x faster than
  LangChain, 7.7x faster than Pydantic AI. (It also edges out the raw
  `ollama` client, but see the note above — that gap is probably measurement
  noise, not a real advantage.)
- Lowest framework overhead of any framework tested on all three tasks, with
  the tightest p95 on completion.

**Ties**

- Statistically tied with hand-written raw HTTP on all three tasks
  (within 0.5 ms at p50). The abstraction is close to free at runtime.
- Tied with LangChain and LlamaIndex on LOC for completion and structured
  output.

**Loses**

- Tool registration: 20 LOC vs LangChain's 10, and worst-in-class total LOC
  (33) among the frameworks. Caused by mandatory hand-written JSON Schema in
  released 0.5.3. Annotation-based inference has landed on `main` but is not
  in a release yet.
- Released 0.5.3 silently accepts a schema-less tool registration that then
  fails at model call time. Also fixed on `main`, also unreleased.
- Feature scope not measured here at all: LangChain and LlamaIndex ship
  retrieval, document loaders, and vector-store integrations that `actants`
  does not have. If you need those, none of the numbers above are the
  deciding factor.

**What this benchmark does not tell you**

Single machine, single model, single OS, small sample counts. No multi-turn
conversation, no streaming throughput, no concurrency or load behaviour, no
retrieval, no remote-provider latency, no memory under sustained use. A 5 ms
per-call difference is irrelevant next to a 150 ms model call unless you are
running at high volume or fanning out many small calls. Choose on features
and ergonomics first; these numbers are a tiebreaker, not a verdict.

---

## Reproducing

Requires [Ollama](https://ollama.com) running locally.

```bash
git clone https://github.com/openintelligence-labs/actants
cd actants
ollama pull qwen2.5:7b
python benchmarks/run_benchmarks.py --runs 7
```

One command does everything: it builds five isolated venvs from the pinned
lockfiles in `benchmarks/requirements/` (default `/tmp/actants-bench-venvs`,
never the actants dev venv), measures footprint, imports, latency, and LOC,
then writes `benchmarks/results.json` and prints the tables.

Useful flags:

```bash
python benchmarks/run_benchmarks.py \
  --model qwen2.5:7b \
  --runs 7 --rounds 7 --import-runs 5 \
  --venv-dir /tmp/actants-bench-venvs \
  --force-venvs \
  --seed 20260730
```

To re-pin a framework to current releases, edit the relevant
`benchmarks/requirements/*.txt` and re-run with `--force-venvs`.

Your absolute numbers will differ — different CPU, different model, different
Ollama build. The *relative* ordering of install size, import time, and
overhead should hold; if it does not on your machine, that is worth an issue.
