"""Minimal structured logger that writes to stdout and a JSONL file."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Logger:
    def __init__(self, log_dir: str | Path, name: str = "run") -> None:
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._t0 = time.time()

    def log(self, step: int, **kwargs: Any) -> None:
        record = {"step": step, "t": round(time.time() - self._t0, 2), **kwargs}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        metrics = " ".join(f"{k}={_fmt(v)}" for k, v in kwargs.items())
        print(f"[{step:>7d}] {metrics}")

    def close(self) -> None:
        self._fh.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
