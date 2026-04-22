"""DPO dataset construction.

Row shape — TRL's standard preference format:

    {
      "prompt":   "...",            # user prompt (string, or messages list)
      "chosen":   "...",            # preferred response
      "rejected": "...",            # dispreferred response
    }

If ``prompt`` is a list of messages, the chat template is applied; otherwise
a single user turn is constructed automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset

from llm_train.data.loading import load_raw_dataset
from llm_train.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from llm_train.config import DataConfig

log = get_logger(__name__)

_REQUIRED = {"prompt", "chosen", "rejected"}


def _normalize_row(
    row: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, str]:
    missing = _REQUIRED - set(row)
    if missing:
        raise ValueError(f"DPO row missing keys: {sorted(missing)}")

    prompt = row["prompt"]
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    elif not isinstance(prompt, str):
        raise TypeError(f"prompt must be str or list[dict], got {type(prompt)}")

    return {
        "prompt": prompt,
        "chosen": row["chosen"] if isinstance(row["chosen"], str) else str(row["chosen"]),
        "rejected": row["rejected"] if isinstance(row["rejected"], str) else str(row["rejected"]),
    }


def build_dpo_datasets(
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset | None]:
    """Build DPO-formatted datasets.

    TRL's ``DPOTrainer`` performs its own tokenization — we just normalize
    the row shape into (prompt, chosen, rejected) strings.
    """
    train = load_raw_dataset(data_cfg, "train")
    evald = (
        load_raw_dataset(data_cfg, "eval")
        if (data_cfg.eval_path or data_cfg.format == "hf")
        else None
    )

    def _map(ds: Dataset) -> Dataset:
        return ds.map(
            _normalize_row,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=[c for c in ds.column_names if c not in _REQUIRED],
            num_proc=max(data_cfg.num_workers, 1),
            desc="Normalizing (DPO)",
        )

    return _map(train), (_map(evald) if evald is not None else None)
