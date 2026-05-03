from __future__ import annotations

import functools
import json as _json
import sys
from typing import Any

try:
    import click
    from rich.console import Console
except ImportError as exc:
    raise ImportError("actants.cli requires the 'cli' extra: pip install 'actants[cli]'") from exc

from actants.observability.logging import setup_logging

console = Console(stderr=False)
_err_console = Console(stderr=True)


def make_app(name: str, *, help: str | None = None) -> click.Group:  # noqa: A002 — matches Click idiom
    """Create a Click group with sensible defaults for an actants-based app.

    Use as the top-level command group for your CLI::

        app = make_app("deepdive", help="Local research agent")

        @app.command()
        @common_options
        def search(query: str): ...
    """

    @click.group(
        name=name,
        help=help,
        context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
    )
    def app() -> None:
        pass

    return app


def common_options(func):
    """Apply standard --debug, --quiet, --json, --log-format flags to a Click command.

    Reads them, configures logging, and pops them from kwargs before calling the wrapped
    function. The function still receives a ``json_output: bool`` kwarg if it asks for it.
    """

    @click.option("--debug/--no-debug", default=False, help="Verbose logging.")
    @click.option("--quiet/--no-quiet", default=False, help="Suppress non-error output.")
    @click.option("--json", "json_output", is_flag=True, default=False, help="JSON output.")
    @click.option(
        "--log-format",
        type=click.Choice(["pretty", "json"]),
        default="pretty",
        help="Log rendering format.",
    )
    @functools.wraps(func)
    def wrapper(*args, debug: bool, quiet: bool, json_output: bool, log_format: str, **kwargs):
        level = "debug" if debug else "error" if quiet else "info"
        setup_logging(level=level, format=log_format)  # type: ignore[arg-type]
        if "json_output" in func.__code__.co_varnames:
            kwargs["json_output"] = json_output
        return func(*args, **kwargs)

    return wrapper


def success(message: str) -> None:
    console.print(f"[bold green]✓[/] {message}")


def error(message: str, *, exit_code: int = 1) -> None:
    _err_console.print(f"[bold red]✗[/] {message}")
    sys.exit(exit_code)


def emit_json(data: Any) -> None:
    """Write JSON to stdout (no styling). Use when --json is set."""
    sys.stdout.write(_json.dumps(data, default=str, indent=2) + "\n")
    sys.stdout.flush()
