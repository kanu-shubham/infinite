#!/usr/bin/env python
"""Entrypoint for ``accelerate launch scripts/run_distill.py --config ...``."""

from __future__ import annotations

import argparse

from llm_train.config import DistillConfig, load_config
from llm_train.training.distill import run_distill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, DistillConfig):
        raise SystemExit(f"Expected task=distill, got task={cfg.task}")
    run_distill(cfg, source_config_path=args.config)


if __name__ == "__main__":
    main()
