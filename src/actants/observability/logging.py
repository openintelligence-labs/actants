from __future__ import annotations

import logging
import sys
from typing import Any, Literal, TextIO

import structlog
from structlog.typing import Processor

LogFormat = Literal["pretty", "json"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


def setup_logging(
    *,
    level: LogLevel = "info",
    format: LogFormat = "pretty",  # noqa: A002 — matches stdlib logging idiom
    stream: TextIO | None = None,
) -> None:
    """Configure structlog + stdlib logging in one call.

    Idempotent — safe to call multiple times. Apps should call this once at startup
    (e.g. in their CLI entrypoint). ``pretty`` uses ConsoleRenderer with colors;
    ``json`` emits one JSON object per line for log aggregation.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        stream=stream or sys.stderr,
        force=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a structlog BoundLogger. Pass ``__name__`` to scope to your module."""
    return structlog.get_logger(name) if name else structlog.get_logger()
