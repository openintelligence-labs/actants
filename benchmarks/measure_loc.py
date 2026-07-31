"""Count lines of code and imports for each task, from the real task files.

Counting rules, applied identically to every framework:

* Only lines between the ``# LOC_A_START`` / ``# LOC_A_END`` markers count.
* Blank lines and comment-only lines are excluded.
* ``import`` / ``from ... import`` lines count as both LOC and imports.
* A task's total includes the shared helpers it actually calls (for example
  pydantic-ai's ``_model()`` factory), because a user would have to write
  those too.
* The ``Person`` model and the tool function body are counted for every
  framework, since all of them need one.

This is a proxy for "how much do I have to write and understand", not a
quality judgement. Fewer lines is not automatically better; the doc reports
the snippets so readers can judge for themselves.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TASKS = ("A", "B", "C")

# Helper functions defined outside a task block but called from inside it.
# Their bodies are attributed to every task that calls them.
SHARED_HELPERS = {"_model", "_build_tools"}


def _blocks(source: str) -> dict[str, list[str]]:
    """Split a task file into its marked A/B/C regions."""
    out: dict[str, list[str]] = {}
    lines = source.splitlines()
    for task in TASKS:
        start = end = None
        for i, line in enumerate(lines):
            if line.strip() == f"# LOC_{task}_START":
                start = i + 1
            elif line.strip() == f"# LOC_{task}_END":
                end = i
        if start is not None and end is not None:
            out[task] = lines[start:end]
    return out


def _is_code(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _helper_sizes(source: str) -> dict[str, int]:
    """Measure each shared helper's line count so callers can be charged for it."""
    tree = ast.parse(source)
    sizes: dict[str, int] = {}
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in SHARED_HELPERS:
            body = lines[node.lineno - 1 : node.end_lineno]
            sizes[node.name] = sum(1 for line in body if _is_code(line))
    return sizes


def analyse(path: Path) -> dict:
    source = path.read_text()
    helper_sizes = _helper_sizes(source)
    blocks = _blocks(source)

    result: dict[str, dict] = {}
    for task, lines in blocks.items():
        code = [line for line in lines if _is_code(line)]
        loc = len(code)
        imports = sum(
            1 for line in code if line.startswith(("import ", "from ")) or " import " in line
        )
        # A helper defined *inside* this block is already counted; only add
        # helpers defined elsewhere in the file that this block calls.
        block_text = "\n".join(lines)
        for helper, size in helper_sizes.items():
            if f"{helper}(" in block_text and f"def {helper}(" not in block_text:
                loc += size
        result[task] = {"loc": loc, "imports": imports}
    return result


def main() -> None:
    tasks_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "tasks"
    out = {}
    for path in sorted(tasks_dir.glob("task_*.py")):
        out[path.stem.removeprefix("task_")] = analyse(path)
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
