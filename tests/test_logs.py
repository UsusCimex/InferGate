"""Root logging in text or JSON lines."""
from __future__ import annotations

import json
import logging
import sys

import pytest

from app.monitoring import set_request_id
from app.monitoring.logs import JsonFormatter, configure_logging


@pytest.fixture
def restore_root():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


def test_json_line_carries_level_logger_request_id_and_extras():
    set_request_id("req-7")
    try:
        record = logging.makeLogRecord(
            {"name": "app.test", "levelno": logging.WARNING, "levelname": "WARNING",
             "msg": "loaded %s", "args": ("flux",), "model_id": "flux"}
        )
        line = json.loads(JsonFormatter().format(record))
    finally:
        set_request_id(None)
    assert line["level"] == "warning"
    assert line["logger"] == "app.test"
    assert line["msg"] == "loaded flux"
    assert line["request_id"] == "req-7"
    assert line["model_id"] == "flux"


def test_json_line_keeps_the_traceback():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.makeLogRecord({"msg": "failed", "exc_info": sys.exc_info()})
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(record))["exc"]


def test_configure_logging_switches_to_json(monkeypatch, restore_root):
    monkeypatch.setenv("INFERGATE_LOG_JSON", "true")
    configure_logging(logging.INFO)
    assert isinstance(restore_root.handlers[0].formatter, JsonFormatter)
    assert logging.getLogger("uvicorn.error").propagate is True
