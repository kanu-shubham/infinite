from __future__ import annotations

import pytest
import torch

from llm_train.training.collators import IGNORE_INDEX, CausalCollator


class _FakeTok:
    pad_token_id = 0


def test_collator_pads_and_masks_labels() -> None:
    c = CausalCollator(tokenizer=_FakeTok(), pad_to_multiple_of=8)
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [IGNORE_INDEX, 5]},
    ]
    batch = c(features)
    assert batch["input_ids"].shape == (2, 8)
    # padding never contributes to loss
    assert (batch["labels"][:, 3:] == IGNORE_INDEX).all()
    # attention mask zeros on padding
    assert batch["attention_mask"][0, 3:].sum() == 0


def test_collator_requires_pad_token() -> None:
    class _NoPad:
        pad_token_id = None

    c = CausalCollator(tokenizer=_NoPad(), pad_to_multiple_of=None)
    with pytest.raises(RuntimeError):
        c([{"input_ids": [1], "attention_mask": [1], "labels": [1]}])


def test_collator_returns_long_tensors() -> None:
    c = CausalCollator(tokenizer=_FakeTok(), pad_to_multiple_of=None)
    batch = c([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}])
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
