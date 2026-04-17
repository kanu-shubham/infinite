from __future__ import annotations

from dataclasses import dataclass

import torch

from lora_finetune.data.collator import CausalCollator
from lora_finetune.data.constants import IGNORE_INDEX


@dataclass
class FakeTok:
    pad_token_id: int = 0
    eos_token_id: int = 0


def test_pads_to_multiple_of_8_and_masks_labels() -> None:
    collator = CausalCollator(tokenizer=FakeTok(), pad_to_multiple_of=8)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ]
    )
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    # Padding positions in labels must be IGNORE_INDEX.
    assert batch["labels"][0, 3:].tolist() == [IGNORE_INDEX] * 5
    assert batch["labels"][1, 2:].tolist() == [IGNORE_INDEX] * 6
    # Attention mask zeros on padded positions.
    assert batch["attention_mask"][0, 3:].tolist() == [0] * 5
    assert torch.equal(batch["input_ids"][0, :3], torch.tensor([1, 2, 3]))
