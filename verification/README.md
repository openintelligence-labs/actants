# Live provider verification harness

Runs each actants provider against a **real** endpoint and reports, per check, whether
the wire format actants believes in is the one the provider actually speaks.

This is an operational script, not part of the shipped package. It lives outside `src/`,
so it is excluded from the wheel and from `mypy --strict src/`.

## Running

```bash
python -m verification.run                    # free providers only (Ollama). No paid calls.
python -m verification.run --yes              # every provider with a key in the environment
python -m verification.run --only openai --yes
python -m verification.run --compat-probe     # also exercise the OpenAI-compatible class locally
python -m verification.run --json results.json
```

Keys are read from the environment and never printed. A provider with no key **skips**;
it is never a failure, because the harness has to be useful to someone holding exactly
one key.

## Spend guard

Paid providers are not called without `--yes`. Every run prints a pessimistic upper-bound
estimate first, uses the cheapest model each provider sells, caps `max_tokens` at 64, and
sends single-sentence prompts. A full seven-check run against one provider is a fraction
of a cent.

## Statuses

| Status | Meaning |
|---|---|
| `pass` | The check ran against the live API and asserted successfully. |
| `fail` | The check ran and actants behaved wrongly. A genuine finding. |
| `skip` | Not attempted — no key, or the provider declares it cannot do this. |
| `blocked` | The account refused the call (no credits, revoked key). The integration was never exercised, so this is neither a pass nor evidence of a defect. |

The `blocked` status exists because an exhausted billing balance would otherwise be
reported as a wall of `FAIL`, wrongly implying actants is broken against that provider.

## Adding a provider

Add a `ProviderTarget` row to `providers.py` with the cheapest model that provider
sells. `_assert_table_covers_actants()` fails at import if actants gains a provider with
no row here, so a new provider cannot become silently unverified *and* silently
unreported.

Results are recorded in [`../docs/PROVIDER_VERIFICATION.md`](../docs/PROVIDER_VERIFICATION.md).
