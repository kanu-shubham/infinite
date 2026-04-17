"""Structured JSON logging for training and serving."""
from __future__ import annotations

import logging
import os
import sys

from pythonjsonlogger import jsonlogger


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level"},
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(lvl)
    for noisy in ("urllib3", "filelock", "accelerate.utils.other"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
