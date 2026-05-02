"""``python -m actants.bench`` entrypoint.

Outputs a Markdown comparison table. Frameworks not installed are skipped
silently (the table just shows ``not installed``).
"""

from __future__ import annotations

import argparse
import sys

from actants.bench.runner import COMPETITORS, format_table, run_all


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m actants.bench")
    parser.add_argument("--samples", type=int, default=3, help="Subprocess samples per measurement")
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Comma-separated framework names. Default: all known competitors.",
    )
    args = parser.parse_args()

    targets = COMPETITORS
    if args.compare:
        wanted = {n.strip() for n in args.compare.split(",")}
        targets = [f for f in COMPETITORS if f.name in wanted]
        unknown = wanted - {f.name for f in COMPETITORS}
        if unknown:
            print(f"# unknown frameworks ignored: {sorted(unknown)}", file=sys.stderr)

    results = run_all(targets, samples=args.samples)
    print(format_table(results))


if __name__ == "__main__":
    main()
