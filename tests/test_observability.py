from __future__ import annotations

import io
import json

import structlog

from agentic_kit.observability import get_logger, setup_logging


def test_setup_logging_pretty_renders_to_stream():
    buf = io.StringIO()
    setup_logging(level="info", format="pretty", stream=buf)
    log = get_logger("test")
    log.info("hello", key="value")
    output = buf.getvalue()
    assert "hello" in output
    assert "key" in output


def test_setup_logging_json_emits_json_line():
    buf = io.StringIO()
    setup_logging(level="info", format="json", stream=buf)
    log = get_logger("test")
    log.info("structured", count=3)
    line = buf.getvalue().strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["event"] == "structured"
    assert parsed["count"] == 3


def test_setup_logging_respects_level():
    buf = io.StringIO()
    setup_logging(level="error", format="json", stream=buf)
    log = get_logger("test")
    log.info("should-not-appear")
    log.error("should-appear")
    out = buf.getvalue()
    assert "should-appear" in out
    assert "should-not-appear" not in out


def test_setup_logging_idempotent():
    buf = io.StringIO()
    setup_logging(level="info", format="json", stream=buf)
    setup_logging(level="info", format="json", stream=buf)
    # No exception, structlog still works
    structlog.get_logger().info("ok")
