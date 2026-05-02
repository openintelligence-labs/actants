"""Build a CLI app in <30 lines using agentic_kit.cli helpers.

Run::

    pip install -e ".[cli]"
    python examples/12_cli_app.py greet --name World
    python examples/12_cli_app.py greet --name World --json
"""

from __future__ import annotations

import click

from agentic_kit.cli import common_options, console, make_app, success
from agentic_kit.cli.app import emit_json

app = make_app("greeter", help="Demo CLI built on agentic-kit")


@app.command()
@click.option("--name", default="world")
@common_options
def greet(name: str, json_output: bool) -> None:
    """Say hello."""
    if json_output:
        emit_json({"greeting": f"Hello, {name}!"})
    else:
        console.print(f"[bold cyan]Hello, {name}![/]")
        success("done")


if __name__ == "__main__":
    app()
