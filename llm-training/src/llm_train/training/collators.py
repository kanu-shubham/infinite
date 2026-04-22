"""Data collators for causal-LM training with label masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


@dataclass
class CausalCollator:
    """Pad ``input_ids``, ``attention_mask``, and ``labels`` to the longest
    sequence in the batch. ``labels`` pads with ``-100`` so padding tokens
    never contribute to the loss.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id.")

        input_ids, attn, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def first_batch_preview(batch: dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode the first example of a batch. Useful for logging sanity checks."""
    ids = batch["input_ids"][0].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


__all__: list[Any] = ["CausalCollator", "first_batch_preview", "IGNORE_INDEX"]
