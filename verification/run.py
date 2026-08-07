"""Live provider verification harness.

Runs every actants provider for which an API key is present against a real endpoint and
reports, per check, whether the wire format actants believes in is the one the provider
actually speaks. Providers with no key SKIP; they are never a failure, because the
harness has to be useful to someone holding exactly one key.

    python -m verification.run                    # free providers only (Ollama)
    python -m verification.run --yes              # everything with a key present
    python -m verification.run --only openai --yes
    python -m verification.run --json out.json

Paid calls never happen without ``--yes``. The estimate printed before the run is
deliberately pessimistic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from actants.llm.client import LLM, LLMSettings
from actants.llm.openai_compatible import openai_compatible_provider
from verification.checks import (
    CheckResult,
    check_complete,
    check_cost_attribution,
    check_stream,
    check_stream_matches_complete,
    check_streaming_tool_call,
    check_structured_output,
    check_tool_call,
)
from verification.providers import COMPAT_PROBE, TARGETS, ProviderTarget

#: Worst case tokens one provider's full run sends and receives. Seven completions of a
#: ~60-token prompt with max_tokens=64, plus the agent loop's extra round-trips.
_EST_TOKENS_PER_PROVIDER = 3000

_GREEN, _RED, _YELLOW, _DIM, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _colored(status: str) -> str:
    if not sys.stdout.isatty():
        return status.upper()
    color = {"pass": _GREEN, "fail": _RED, "skip": _YELLOW, "blocked": _YELLOW}.get(status, "")
    return f"{color}{status.upper()}{_RESET}"


def _build_client(target: ProviderTarget) -> LLM:
    """Build the client for one target, including the generated compatible-provider probe."""
    if target.base_url is not None:
        provider = openai_compatible_provider(target.name, target.base_url, "local probe")(
            api_key="local"
        )
        return LLM(provider=provider, model=target.model)
    return LLM(settings=LLMSettings(provider=target.name, model=target.model))


async def run_provider(target: ProviderTarget) -> dict[str, Any]:
    """Run every check against one provider, returning its summary record."""
    print(f"\n=== {target.name} ({target.model}) ===")

    try:
        llm = _build_client(target)
    except Exception as exc:
        print(f"  build client            {_colored('fail')}  {type(exc).__name__}: {exc}")
        return {
            "provider": target.name,
            "model": target.model,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "checks": {},
        }

    declared = {
        "native_schema_mode": llm.provider.native_schema_mode,
        "supports_tool_calls": llm.provider.supports_tool_calls,
        "supports_streaming_tools": llm.provider.supports_streaming_tools,
    }

    results: list[CheckResult] = [
        await check_complete(llm, target.model),
        await check_stream(llm, target.model),
        await check_stream_matches_complete(llm, target.model),
        await check_tool_call(llm, target.model),
        await check_streaming_tool_call(llm, target.model),
        await check_structured_output(llm, target.model, target.expected_schema_mode),
        await check_cost_attribution(llm, target.model),
    ]

    for r in results:
        print(f"  {r.name:<24}{_colored(r.status)}  {_DIM}{r.detail}{_RESET}")

    failed = [r.name for r in results if r.status == "fail"]
    blocked = [r.name for r in results if r.status == "blocked"]
    structured = next(r for r in results if r.name == "structured_output")

    # A provider whose every check was refused at the account level is reported as
    # `blocked`, not `fail`: nothing about actants was exercised, so calling it a
    # failure would be as dishonest as calling it verified.
    if blocked and not failed:
        status = "blocked"
    elif failed:
        status = "fail"
    else:
        status = "pass"

    return {
        "provider": target.name,
        "model": target.model,
        "status": status,
        "blocked_checks": blocked,
        "declared": declared,
        "failed_checks": failed,
        "schema_path": ("native" if structured.data.get("native") else "prompt")
        if structured.status != "skip"
        else None,
        "schema_mode": structured.data.get("mode"),
        "checks": {r.name: {"status": r.status, "detail": r.detail, **r.data} for r in results},
    }


def _partition(
    only: list[str] | None, *, compat_probe: bool
) -> tuple[list[ProviderTarget], list[ProviderTarget]]:
    """Split the table into runnable targets and those skipped for a missing key."""
    table = [*TARGETS, COMPAT_PROBE] if compat_probe else list(TARGETS)
    selected = [t for t in table if not only or t.name in only]
    runnable = [t for t in selected if t.key_present()]
    skipped = [t for t in selected if not t.key_present()]
    return runnable, skipped


def _estimate(targets: list[ProviderTarget]) -> float:
    return sum(t.approx_price_per_1m * (_EST_TOKENS_PER_PROVIDER / 1_000_000) for t in targets)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="actually call paid APIs")
    parser.add_argument("--only", nargs="+", metavar="PROVIDER", help="restrict to these providers")
    parser.add_argument("--json", metavar="PATH", help="write the machine-readable summary here")
    parser.add_argument(
        "--compat-probe",
        action="store_true",
        help="also exercise the generated OpenAI-compatible provider class against a "
        "local Ollama /v1 endpoint (free; verifies the shared request path only)",
    )
    args = parser.parse_args()

    runnable, no_key = _partition(args.only, compat_probe=args.compat_probe)
    paid = [t for t in runnable if not t.free]
    free = [t for t in runnable if t.free]

    print("actants live provider verification")
    print(f"  free providers:    {', '.join(t.name for t in free) or '(none)'}")
    print(f"  keyed providers:   {', '.join(t.name for t in paid) or '(none)'}")
    print(f"  skipped (no key):  {', '.join(t.name for t in no_key) or '(none)'}")
    print(f"  estimated spend:   ~${_estimate(paid):.4f} (pessimistic upper bound)")

    if paid and not args.yes:
        print(
            f"\n{_YELLOW}Paid providers will NOT be called without --yes.{_RESET} "
            "Re-run with --yes to spend the estimate above."
        )
        paid = []

    to_run = free + paid
    if not to_run:
        print("\nNothing to run.")
        return 0

    records = [await run_provider(t) for t in to_run]
    for t in no_key:
        records.append(
            {
                "provider": t.name,
                "model": t.model,
                "status": "skip",
                "reason": f"{t.env_var} not set",
                "checks": {},
            }
        )
    for t in (t for t in runnable if not t.free and not args.yes):
        records.append(
            {
                "provider": t.name,
                "model": t.model,
                "status": "skip",
                "reason": "paid provider, --yes not given",
                "checks": {},
            }
        )

    print("\n=== summary ===")
    for rec in records:
        line = f"  {rec['provider']:<32}{_colored(str(rec['status']))}"
        if rec["status"] == "skip":
            line += f"  {_DIM}{rec.get('reason', '')}{_RESET}"
        elif rec["status"] == "blocked":
            line += f"  {_DIM}account refused the call; integration not exercised{_RESET}"
        elif rec.get("failed_checks"):
            line += f"  {_DIM}failed: {', '.join(rec['failed_checks'])}{_RESET}"
        elif rec.get("schema_path"):
            line += f"  {_DIM}schema path: {rec['schema_path']}{_RESET}"
        print(line)

    summary = {
        "verified": [r["provider"] for r in records if r["status"] == "pass"],
        "failed": [r["provider"] for r in records if r["status"] == "fail"],
        "blocked": [r["provider"] for r in records if r["status"] == "blocked"],
        "skipped": [r["provider"] for r in records if r["status"] == "skip"],
        "providers": records,
    }
    payload = json.dumps(summary, indent=2, default=str)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(payload + "\n")
        print(f"\nwrote {args.json}")
    else:
        print("\n" + payload)

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    # Never inherit a key the caller did not mean to spend: the harness reads only the
    # environment it was launched with, and prints nothing from it.
    if os.environ.get("ACTANTS_API_KEY"):
        print("Refusing to run with ACTANTS_API_KEY set; it would override every provider.")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main()))
