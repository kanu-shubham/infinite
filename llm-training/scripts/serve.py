#!/usr/bin/env python
"""Serve a model via FastAPI. Thin wrapper over ``llm_train.cli serve``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.environ["LLM_TRAIN_MODEL_PATH"] = args.model
    uvicorn.run(
        "llm_train.inference.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
