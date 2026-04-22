"""Distillation dataset construction.

Input shapes:

- ``{"messages": [...]}`` or ``{"prompt": "...", "response": "..."}``
  (same as SFT — used for both teacher-forcing and sequence-level distill)

- ``{"prompt": "..."}`` (unlabeled)
  Usable only for sequence-level distillation where the teacher generates
  targets on the fly.

For **logit distillation** we tokenize identically to SFT so the student
sees the same input_ids that will be fed to the teacher during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset

from llm_train.data.sft import build_sft_datasets

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from llm_train.config import DataConfig, TokenizerConfig


def build_distill_datasets(
    data_cfg: DataConfig,
    tok_cfg: TokenizerConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset | None]:
    """Reuses the SFT tokenizer pipeline — the teacher sees the same inputs."""
    return build_sft_datasets(
        data_cfg, tok_cfg, tokenizer, completion_only_loss=True
    )
