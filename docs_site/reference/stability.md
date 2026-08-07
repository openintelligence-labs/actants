# Stability policy

actants 1.0 is a commitment about **what will not break**. This page states exactly what
that covers, what it does not, and how anything that must change gets changed.

## The guarantee in one sentence

Within the 1.x series, code that uses only the public API and does not depend on
provisional surfaces will keep working, and will keep meaning the same thing.

"Keep meaning the same thing" is the load-bearing half. A change that leaves your code
importing and running but silently returning different numbers — a cost that stops being
counted, a cache that starts serving a different entry — is a breaking change here, and is
treated as one.

## What counts as the public API

**The public API is exactly what `actants.__all__` exports**, plus the documented
attributes and methods of those objects.

```python
import actants

print(actants.__all__)  # the complete, authoritative list
```

If you can reach it as `from actants import X`, it is covered. Three qualifications:

- **Names beginning with `_` are private**, wherever they live, including on public
  classes. `LLM._stream_layered` and `SqliteVecCache._connect` may change or vanish in
  any release.
- **Submodule paths are covered only for the symbols re-exported at top level.**
  `from actants import CompletionResult` is stable; `from actants.llm.base import
  CompletionResult` happens to work today, and the module layout is not itself a promise.
  Import from the top level.
- **Subclassing `BaseLLMProvider` is supported and covered.** Its three methods —
  `complete`, `stream_events`, `health` — are the provider contract, and adding a required
  parameter to any of them would be a major-version change. Providers may gain *optional*
  keyword parameters in a minor release, so implementations should accept `**kwargs`, as
  every built-in provider does.

## What semver means here

- **Patch** (1.0.x) — bug fixes, performance, docs, added tests. No API change. A fix that
  corrects a *wrong result* ships here even though the output changes, because the old
  output was a defect; such fixes are always called out in the changelog.
- **Minor** (1.x.0) — new modules, new functions, new optional parameters, new providers,
  new pricing entries, new canonical values added to open-ended enumerations (see below).
  Existing calls keep working.
- **Major** (2.0.0) — anything else: removing or renaming a public name, changing a
  parameter's meaning, narrowing a type, changing a default that alters results.

### Specifically permitted in a minor release

These are worth stating because they can look breaking:

- **Adding a field to a pydantic model** such as `CompletionResult`. Code that constructs
  these positionally or asserts on an exact `model_dump()` may notice; code that reads
  fields will not.
- **Adding a member to `FinishReason`.** The set of canonical finish reasons is
  **open for extension**. A provider inventing a genuinely new category of stop reason may
  get a new canonical value in a minor release, so treat `finish_reason` as an open set:
  handle the values you care about and give the rest a default branch. Values are never
  *removed* or *repurposed* in 1.x, and an unrecognized provider string always maps to
  `"unknown"` rather than raising.
- **Correcting the pricing table.** Prices are facts about the outside world, not API. A
  wrong price is a bug and gets fixed in a patch release; your reported costs will change,
  because they were wrong before.
- **Adding an optional keyword parameter** to a public function.

## Provisional surfaces

These ship in 1.0 and are genuinely useful, but are **not** covered by the compatibility
guarantee — they may change in a minor release. Each is marked as provisional in its own
documentation as well.

| Surface | Why it is provisional |
| --- | --- |
| `actants.a2a` | Tracks the A2A protocol spec, which is still moving. |
| `actants.mcp` | Tracks the MCP spec and the upstream `mcp` SDK's own pre-1.0 API. |
| `actants.bench` | The benchmark harness and its result schema are still being shaped by what turns out to be worth measuring. |
| `SqliteVecCache` on-disk format | The database records a schema version and *discards* an incompatible file rather than misreading it. The file format may change in any release; the cache is a disposable accelerator, never a store of record. |
| OpenTelemetry span and attribute names | Follow the OTel GenAI semantic conventions, which are themselves not yet stable. |

Everything else in `__all__` is covered.

## What is explicitly *not* promised

- **Model output.** actants does not control what an LLM says. Nothing about response
  content is stable, and no test should assert on it.
- **Costs and token counts** for a given prompt. Providers change tokenizers and prices.
- **Cache hit rates.** Semantic similarity is a heuristic; threshold behaviour may be
  tuned within 1.x.
- **`repr()` and log output.** Useful for humans, not for parsing.
- **Exception *messages*.** The exception *types* in `__all__` are stable and safe to
  catch; the wording inside them is improved freely.
- **Python versions.** Support follows the [SPEC 0](https://scientific-python.org/specs/spec-0000/)
  schedule; dropping an end-of-life Python is a minor release, not a major one.

## Deprecation process

Nothing in the public API is removed without warning. The sequence is:

1. **Announce.** The replacement ships first, in a minor release. The old name keeps
   working and starts emitting a `DeprecationWarning` naming what to use instead.
2. **Wait.** At least **two minor releases and six months**, whichever is longer. During
   this window both spellings work.
3. **Remove.** Only in the next major release.

`DeprecationWarning` is silent by default in Python. To see them:

```bash
python -W error::DeprecationWarning -m pytest
```

Running your test suite that way is the supported method for finding out whether an
actants upgrade will affect you before it does.

A deprecation is never used to change a name's *meaning*. If behaviour must change
incompatibly, the new behaviour gets a new name and the old one is deprecated — you will
never find that the same call quietly started doing something different.

## Security

Security fixes ship as fast as they are ready, to the current minor series. If a fix is
unavoidably breaking, it still ships, and the changelog says so plainly. See
[SECURITY.md](https://github.com/openintelligence-labs/actants/blob/main/SECURITY.md)
for how to report a vulnerability.

## How this is enforced

The policy is checked, not merely promised:

- `actants.__all__` and its type-checker re-exports are pinned by tests, so a public name
  cannot be added or dropped unnoticed.
- `mypy --strict` runs against the whole of `src/` in CI, not just the public entry
  point, so the types downstream consumers see are the types they actually get. The one
  exception is the `sqlite_vec` import in the semantic cache: that package ships no
  `py.typed` marker and has no stub package, so its import alone is excused by a scoped
  override in `pyproject.toml`. The module importing it is still fully checked.
- Every code block in this documentation and in the README is executed as a test, so a
  documented call that stops working fails the build.
- Provider mappings, cost attribution, and cache-key scoping each have dedicated
  regression tests, because those are the places where a break would otherwise be silent.
