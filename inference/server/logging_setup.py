"""Structured JSON logging.

Single-line JSON makes per-request fields (request_id, ttft_ms, etc.)
trivially queryable in Loki/Cloud Logging without regex.
"""
from __future__ import annotations

import logging
import sys
import time

import orjson


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra fields stamped via logger.info(..., extra={...}).
        for k, v in record.__dict__.items():
            if k in payload or k.startswith("_"):
                continue
            if k in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
                "taskName",
            }:
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return orjson.dumps(payload).decode()


def configure(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # vLLM is chatty at INFO; keep it but respect root level.
    logging.getLogger("vllm").setLevel(level.upper())
