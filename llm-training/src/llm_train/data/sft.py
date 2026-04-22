"""SFT dataset construction.

Accepts two row shapes:

1. ``{"messages": [{"role": ..., "content": ...}, ...]}``
   Conversation is rendered via the tokenizer's chat template.

2. ``{"prompt": "...", "response": "..."}``
   Treated as a single user turn + assistant response.

When ``completion_only_loss`` is true, loss is only computed on the assistant
response tokens — this is the default and matches standard SFT practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset

from llm_train.data.loading import load_raw_dataset
from llm_train.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from llm_train.config import DataConfig, TokenizerConfig

log = get_logger(__name__)

IGNORE_INDEX = -100


def _to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in row:
        return row["messages"]
    if "prompt" in row and "response" in row:
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    raise ValueError(
        "SFT row must contain either 'messages' or ('prompt' + 'response'); "
        f"got keys {list(row.keys())}"
    )


def _encode_example(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    completion_only_loss: bool,
) -> dict[str, list[int]]:
    messages = _to_messages(row)

    if completion_only_loss:
        # Render prompt (everything up to the final assistant turn) and the
        # full conversation separately so we can compute where the assistant
        # response starts.
        if messages[-1]["role"] != "assistant":
            raise ValueError("Final message must have role='assistant' for SFT training.")
        prompt_messages = messages[:-1]
        prompt_ids: list[int] = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        # Truncate right; preserve the prompt when possible.
        full_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX
        attention_mask = [1] * len(full_ids)
        return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}

    ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    ids = ids[:max_length]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}


def build_sft_datasets(
    data_cfg: DataConfig,
    tok_cfg: TokenizerConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    completion_only_loss: bool = True,
) -> tuple[Dataset, Dataset | None]:
    """Build tokenized train (and optional eval) datasets for SFT."""
    train_raw = load_raw_dataset(data_cfg, "train")
    eval_raw = (
        load_raw_dataset(data_cfg, "eval")
        if (data_cfg.eval_path or data_cfg.format == "hf")
        else None
    )

    def _map(ds: Dataset) -> Dataset:
        return ds.map(
            _encode_example,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": tok_cfg.model_max_length,
                "completion_only_loss": completion_only_loss,
            },
            remove_columns=ds.column_names,
            num_proc=max(data_cfg.num_workers, 1),
            desc="Tokenizing (SFT)",
        )

    train = _map(train_raw)
    evald = _map(eval_raw) if eval_raw is not None else None
    log.info("SFT train tokens ~ %d", sum(len(r["input_ids"]) for r in train.select(range(min(100, len(train))))) * max(len(train) // 100, 1))
    return train, evald
