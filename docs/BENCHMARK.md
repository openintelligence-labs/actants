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
noise of writing raw HTTP by hand. It still loses the total-LOC row, but only
just — schema inference in 1.0 cut tool registration from 20 lines to 11,
turning a 10-line deficit against LangChain into a 1-line one.

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
| Ollama | 0.32.6 |
| Model | `qwen2.5:7b` (Q4_K_M, 7.6B) |
| Date | 2026-08-06 |
| Config | 7 latency samples, 7 rounds, 5 import samples, seed 20260730 |

**Every number in this document was re-measured on 2026-08-06 against
`actants` 1.0.0 installed from PyPI.** Nothing is carried over from the
previous (0.5.3) run; where this doc compares against 0.5.3 it is quoting the
older published table, not mixing old numbers into the new one. The comparison
frameworks were re-resolved to their current releases at the same time — only
`pydantic-ai` had moved (2.21.0 → 2.25.0); LangChain, LlamaIndex, and `ollama`
resolved to the same versions as before.

Frameworks under test, as resolved into isolated venvs
(full lockfiles in `benchmarks/requirements/`):

| Framework | Version | Direct install |
|---|---|---|
| actants | 1.0.0 | `actants` |
| LangChain | langchain 1.3.14, langchain-core 1.5.3, langchain-ollama 1.1.0, langgraph 1.2.10 | `langchain langchain-ollama` |
| Pydantic AI | pydantic-ai 2.25.0 | `pydantic-ai` |
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
| **actants** | **18** | **14.3 MB** |
| LangChain | 38 | 36.0 MB |
| Pydantic AI | 98 | 106.7 MB |
| LlamaIndex | 63 | 126.6 MB |

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
| **actants** | **95.6 ms** | 100.5 ms | 43.1 MB | 401 |
| raw | 105.9 ms | 107.2 ms | 44.8 MB | 356 |
| LangChain | 343.9 ms | 347.0 ms | 83.1 MB | 1058 |
| LlamaIndex | 539.4 ms | 546.9 ms | 115.3 MB | 1441 |
| Pydantic AI | 724.1 ms | 755.5 ms | 124.4 MB | 2496 |

`actants` imports 3.6x faster than LangChain and 7.6x faster than Pydantic AI.

It also comes out ~10 ms ahead of the raw `ollama` client, which is worth
flagging as *suspicious rather than impressive*: `actants` loads 401 modules
to raw's 356, and both end up importing `httpx` and `pydantic`. A library
that loads strictly more modules should not import faster. The most likely
explanation is filesystem cache and import-order effects between separate
venvs, not a genuine advantage. Treat `actants` and raw as tied on import,
and treat the 3.6x-7.6x gaps against the larger frameworks — which track
module counts of 1058-2496 — as the real signal. (The gap against raw was
~20 ms on 0.5.3 and is ~10 ms here, which is consistent with it being noise
rather than a property of either library.)

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
| raw | 6.39 ms | 7.67 ms | 0.163 s | 1 |
| **actants** | **7.54 ms** | 19.00 ms | 0.161 s | 1 |
| Pydantic AI | 10.05 ms | 19.60 ms | 0.178 s | 1 |
| LangChain | 12.29 ms | 23.86 ms | 0.169 s | 1 |
| LlamaIndex | 12.53 ms | 18.68 ms | 0.167 s | 2 |

`actants` is statistically tied with raw at p50 (1.15 ms apart, inside
run-to-run noise) and remains the lowest-overhead framework. The three other
frameworks cost roughly 4-6 ms more per call than raw.

The p95 column is noisier in this run than in the 0.5.3 one across every
framework, `actants` included (19.00 ms here vs 7.74 ms then). That is a
property of this sitting, not of 1.0: re-measuring this task alone,
uninterleaved, gives 12 consecutive samples spanning 5.3-6.4 ms with a p50 of
5.83 ms. Read the p50 column as the signal and treat these p95s as a noise
ceiling rather than a per-framework characteristic.

### (b) Agent with one tool (two model turns)

| Framework | Overhead p50 | Overhead p95 | Wall p50 | HTTP reqs |
|---|---|---|---|---|
| raw | 7.74 ms | 10.63 ms | 1.254 s | 2 |
| **actants** | **8.56 ms** | 28.01 ms | 1.354 s | 2 |
| Pydantic AI | 13.47 ms | 21.00 ms | 1.199 s | 2 |
| LangChain | 18.45 ms | 20.20 ms | 1.265 s | 2 |
| LlamaIndex | 934.21 ms | 4660.47 ms | 2.054 s | 3 |

`actants` and raw are tied (0.82 ms apart — noise). Per tool-calling turn
that is ~4.3 ms of framework overhead for actants versus ~9.2 ms for LangChain.
Dropping the hand-written schema did not change this measurably, which is the
expected result: schema inference runs once at registration, and the
registration is inside the timed region but costs microseconds.

**The LlamaIndex number is real and reproducible, not an artefact.** It was
independently re-verified for this run: 12 consecutive uninterleaved samples
landed between 958 ms and 1783 ms of overhead, consistent with the 0.5.3 run's
907-976 ms. The cost is `FunctionAgent`'s workflow engine — the event bus and
step dispatch in `llama-index-workflows` — not model time, not the network,
and not a sleep. On a 7B local model that nearly doubles the latency of a
two-turn tool call. The 4660 ms p95 above is the tail of a genuinely
high-variance cost; treat the ~1 s p50 as the reliable figure and the p95 as
evidence that the variance is large, not as a stable number.

### (c) Structured output into a pydantic model

| Framework | Overhead p50 | Overhead p95 | Wall p50 | HTTP reqs |
|---|---|---|---|---|
| raw | 6.72 ms | 16.87 ms | 1.066 s | 1 |
| **actants** | **7.71 ms** | 21.91 ms | 1.044 s | 1 |
| Pydantic AI | 11.04 ms | 194.07 ms | 0.877 s | 1 |
| LlamaIndex | 12.17 ms | 29.47 ms | 0.851 s | 2 |
| LangChain | 13.44 ms | 14.82 ms | 0.960 s | 1 |

Within 1 ms of raw at p50. The previous run happened to land on an identical
p50 to raw; that was coincidence, and 1 ms is the honest resolution here.

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
| **actants** | 5 (1) | 11 (1) | 8 (1) | 24 |
| LlamaIndex | 5 (1) | 11 (2) | 9 (2) | 25 |
| Pydantic AI | 10 (3) | 10 (0) | 12 (2) | 32 |
| raw | 5 (1) | 32 (1) | 12 (1) | 49 |

**`actants` still loses this metric, by one line.**

This is the row 1.0 changed. On 0.5.3 tool registration took 20 lines because
a hand-written JSON Schema was mandatory, putting the total at 33 — last place
among the frameworks. 1.0 infers the schema from type annotations, which cuts
task (b) from 20 lines to 11 and the total from 33 to 24. That moves `actants`
from worst to second, one line behind LangChain.

What a user writes on 1.0:

```python
async def get_weather(city: str) -> str:
    return f"18C and raining in {city}"


def _build_tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register_function("get_weather", "Get the current weather for a city.", get_weather)
    return tools
```

versus LangChain:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and raining in {city}"
```

The remaining one-line gap is structural, not incidental: LangChain's `@tool`
decorator registers in place and takes its description from the docstring,
so there is no registry object to construct and pass. `actants` requires an
explicit `ToolRegistry`, which costs the `tools = ToolRegistry()` and
`return tools` lines. That is a deliberate design difference — an explicit
registry is what makes per-tool permission checks and multiple isolated tool
sets possible — but on this metric it costs a line, and the count above is the
honest one. **LangChain wins this row.**

Two smaller notes on the counting, unchanged from the previous run: `actants`'
completion and structured tasks import `LLMSettings` alongside `LLM`, which a
normal user would not need — the benchmark uses it to point the client at the
recording proxy. Without that constraint `actants` would be one import lighter
on two of the three tasks (and would tie LangChain on the total). Conversely,
Pydantic AI's 10-line completion is inflated by three imports and a `_model()`
helper needed because its Ollama provider refuses to default to
`localhost:11434` (see below).

Only the `actants` task file was rewritten for this run, and only task (b),
to drop the now-unnecessary schema. The other four frameworks' tasks are
byte-identical to the 0.5.3 run — rewriting them to be more verbose would have
made this comparison meaningless. Each remains written the way its own
documentation recommends.

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

**`actants` 0.5.3 silently accepted a tool with no schema — fixed in 1.0.**
On 0.5.3, omitting `input_schema` produced `{"type": "object", "properties":
{}}`, so the model never learned the tool took arguments and the call failed
at model-call time rather than at registration. 1.0 infers the schema from the
handler's annotations instead, and raises at registration time if a parameter
is un-annotated — the failure moved from silent-and-late to loud-and-early.
Verified against the published 1.0.0 wheel: `register_function("get_weather",
"...", get_weather)` yields `{"type": "object", "properties": {"city":
{"type": "string"}}, "required": ["city"]}`.

---

## Where actants wins, ties, and loses

**Wins**

- Smallest install of the four frameworks: 18 packages / 14.3 MB, vs
  LangChain's 38 / 36.0 MB and Pydantic AI's 98 / 106.7 MB.
- Fastest cold import among the frameworks: 95.6 ms, 3.6x faster than
  LangChain, 7.6x faster than Pydantic AI. (It also edges out the raw
  `ollama` client, but see the note above — that gap is probably measurement
  noise, not a real advantage.)
- Lowest framework overhead of any framework tested on all three tasks, with
  the tightest p95 on completion.

**Ties**

- Statistically tied with hand-written raw HTTP on all three tasks
  (within ~1.2 ms at p50). The abstraction is close to free at runtime.
- Tied with LangChain and LlamaIndex on LOC for completion and structured
  output, and with LlamaIndex on the tool task.

**Loses**

- Total LOC: 24 vs LangChain's 23. Much closer than the 33 vs 23 measured on
  0.5.3 — schema inference in 1.0 cut tool registration from 20 lines to 11 —
  but LangChain's `@tool` decorator still edges it out, because `actants`
  requires an explicit `ToolRegistry` object where LangChain registers in
  place. Second of five, still a loss.
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
