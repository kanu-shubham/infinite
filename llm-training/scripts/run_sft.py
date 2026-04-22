#!/usr/bin/env python
"""Entrypoint for ``accelerate launch scripts/run_sft.py --config ...``."""

from __future__ import annotations

import argparse

from llm_train.config import SFTConfig, load_config
from llm_train.training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, SFTConfig):
        raise SystemExit(f"Expected task=sft, got task={cfg.task}")
    run_sft(cfg, source_config_path=args.config)


if __name__ == "__main__":
    main()
