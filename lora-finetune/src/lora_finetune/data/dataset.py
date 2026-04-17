"""Dataset loading, templating and tokenization.

Responsibilities:
  * Load from the HF Hub, local JSONL, or CSV.
  * Apply a prompt template.
  * Tokenize with an assistant-response loss mask so that only the response
    tokens contribute to the loss.
  * Split train/eval when no eval file is provided.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from lora_finetune.config import DataConfig
from lora_finetune.data.constants import IGNORE_INDEX
from lora_finetune.data.templates import format_example
from lora_finetune.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["build_datasets", "IGNORE_INDEX"]


def _load_raw(cfg: DataConfig) -> DatasetDict:
    if cfg.dataset_name:
        logger.info("Loading HF dataset", extra={"dataset": cfg.dataset_name})
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            streaming=cfg.streaming,
        )
        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})
        return ds  # type: ignore[return-value]

    assert cfg.train_file is not None
    ext = Path(cfg.train_file).suffix.lower().lstrip(".")
    loader = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    files = {"train": cfg.train_file}
    if cfg.eval_file:
        files["validation"] = cfg.eval_file
    return load_dataset(loader, data_files=files)  # type: ignore[return-value]


def _extract_fields(example: dict[str, Any], cfg: DataConfig) -> tuple[str, str]:
    if cfg.prompt_field and cfg.response_field:
        return str(example[cfg.prompt_field]), str(example[cfg.response_field])
    # Fallback: assume a single `text_field` already contains prompt+response joined.
    return "", str(example[cfg.text_field])


def _build_tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
):
    max_len = cfg.max_seq_length

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        prompt, response = _extract_fields(example, cfg)

        if prompt:
            prompt_text = format_example(cfg.template, prompt, None, cfg.system_prompt)
            full_text = format_example(cfg.template, prompt, response, cfg.system_prompt)
        else:
            prompt_text = ""
            full_text = response

        full = tokenizer(full_text, truncation=True, max_length=max_len, add_special_tokens=False)
        input_ids = full["input_ids"]
        labels = list(input_ids)

        if prompt_text:
            prompt_ids = tokenizer(
                prompt_text, truncation=True, max_length=max_len, add_special_tokens=False
            )["input_ids"]
            n = min(len(prompt_ids), len(labels))
            for i in range(n):
                labels[i] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return tokenize


def build_datasets(
    cfg: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset | None]:
    """Return (train_dataset, eval_dataset_or_none) ready for a HF Trainer."""
    raw = _load_raw(cfg)

    train_raw = raw["train"]
    eval_raw = raw.get("validation") or raw.get("test")

    if eval_raw is None and cfg.eval_split_ratio > 0:
        split = train_raw.train_test_split(test_size=cfg.eval_split_ratio, seed=42)
        train_raw, eval_raw = split["train"], split["test"]

    tok_fn = _build_tokenize_fn(tokenizer, cfg)
    remove = list(train_raw.column_names)

    logger.info(
        "Tokenizing datasets",
        extra={"train_rows": len(train_raw), "eval_rows": len(eval_raw) if eval_raw else 0},
    )
    train_ds = train_raw.map(tok_fn, remove_columns=remove, num_proc=cfg.num_proc)
    eval_ds = (
        eval_raw.map(tok_fn, remove_columns=remove, num_proc=cfg.num_proc) if eval_raw else None
    )
    return train_ds, eval_ds
