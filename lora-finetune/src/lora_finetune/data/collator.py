"""Data collator that pads variable-length sequences and label masks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from lora_finetune.data.constants import IGNORE_INDEX


@dataclass
class CausalCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        m = self.pad_to_multiple_of
        if m:
            max_len = ((max_len + m - 1) // m) * m
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, attn, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            att = f["attention_mask"]
            lbl = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append(att + [0] * pad)
            labels.append(lbl + [IGNORE_INDEX] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
