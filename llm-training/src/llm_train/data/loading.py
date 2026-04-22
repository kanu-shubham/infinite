"""Dataset loading primitives.

Supports three sources uniformly:
  - jsonl / parquet files on disk
  - Hugging Face Hub datasets

The returned object is always a ``datasets.Dataset`` (single split).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset

from llm_train.utils.logging import get_logger

if TYPE_CHECKING:
    from llm_train.config import DataConfig

log = get_logger(__name__)


def _load_file(path: str, fmt: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    if fmt == "jsonl":
        return load_dataset("json", data_files=str(p), split="train")
    if fmt == "parquet":
        return load_dataset("parquet", data_files=str(p), split="train")
    raise ValueError(f"Unknown file format: {fmt}")


def load_raw_dataset(cfg: DataConfig, split: str) -> Dataset:
    """Load a single split of the configured dataset.

    ``split`` is one of ``"train"`` or ``"eval"``.
    """
    if split not in {"train", "eval"}:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

    if cfg.format == "hf":
        ds_id = cfg.train_path if split == "train" else (cfg.eval_path or cfg.train_path)
        hf_split = cfg.split_train if split == "train" else cfg.split_eval
        ds = load_dataset(ds_id, split=hf_split)
    else:
        path = cfg.train_path if split == "train" else cfg.eval_path
        if path is None:
            raise ValueError(f"No {split} path configured.")
        ds = _load_file(path, cfg.format)

    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    if cfg.shuffle and split == "train":
        ds = ds.shuffle(seed=cfg.seed)

    log.info("Loaded %s split: %d rows", split, len(ds))
    return ds
