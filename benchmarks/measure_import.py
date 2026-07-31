"""Measure cold-import wall time and post-import RSS for one framework.

Run as a subprocess so every sample is a genuinely cold interpreter. Prints a
single JSON object to stdout.

The argument is a Python *statement*, not a bare module name. This matters:
both ``actants`` and ``langchain`` are lazy packages whose top-level
``import`` is a ~0.3ms stub that loads nothing. Timing ``import langchain``
measures the stub, not the framework -- the real cost lands on the first
``from langchain_ollama import ChatOllama``. So each framework is charged for
importing the symbols its benchmark tasks actually use.

Usage: python measure_import.py "from x import y"
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    statement = sys.argv[1]
    start = time.perf_counter()
    exec(compile(statement, "<bench>", "exec"), {})  # noqa: S102
    elapsed = time.perf_counter() - start

    rss_bytes = 0
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; macOS reports bytes.
        rss_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    except ImportError:  # pragma: no cover - non-POSIX
        rss_bytes = 0

    json.dump(
        {
            "statement": statement,
            "import_seconds": elapsed,
            "rss_bytes": rss_bytes,
            "modules_loaded": len(sys.modules),
            "pid": os.getpid(),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
