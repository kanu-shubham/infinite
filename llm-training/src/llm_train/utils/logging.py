"""Structured logging with optional JSON formatting for log shipping."""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

from pythonjsonlogger import jsonlogger

_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    fmt: Literal["text", "json"] | None = None,
) -> None:
    """Configure the root logger once per process.

    Honors ``LOG_LEVEL`` and ``LOG_FORMAT`` env vars if args are omitted.
    Idempotent — safe to call from multiple entrypoints.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = level or os.environ.get("LOG_LEVEL", "INFO")
    fmt = fmt or os.environ.get("LOG_FORMAT", "text")  # type: ignore[assignment]

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers (e.g. transformers sets its own).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "ts", "levelname": "level"},
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Quiet noisy deps by default; callers can override.
    for noisy in ("urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, setting up logging if not yet configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
