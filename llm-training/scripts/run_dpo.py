#!/usr/bin/env python
"""Entrypoint for ``accelerate launch scripts/run_dpo.py --config ...``."""

from __future__ import annotations

import argparse

from llm_train.config import DPOConfig, load_config
from llm_train.training.dpo import run_dpo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, DPOConfig):
        raise SystemExit(f"Expected task=dpo, got task={cfg.task}")
    run_dpo(cfg, source_config_path=args.config)


if __name__ == "__main__":
    main()
