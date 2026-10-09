"""Root logging of the gateway and workers: text lines, or JSON lines with INFERGATE_LOG_JSON=true."""
from __future__ import annotations

import json
import logging
import os

from app.monitoring.request_context import get_request_id

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_RECORD_FIELDS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def json_logs_enabled() -> bool:
    return os.environ.get("INFERGATE_LOG_JSON", "").lower() in {"1", "true", "yes"}


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, message, request id and the `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _RECORD_FIELDS and key not in payload and value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


def configure_logging(level: int) -> None:
    """Route the root logger and uvicorn's loggers through one handler in the chosen format."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_logs_enabled() else logging.Formatter(_TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
