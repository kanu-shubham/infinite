from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["load_base_model", "attach_adapter", "attach_lora", "load_tokenizer", "merge_and_save"]

if TYPE_CHECKING:  # pragma: no cover
    from lora_finetune.models.merge import merge_and_save
    from lora_finetune.models.peft_model import (
        attach_adapter,
        attach_lora,
        load_base_model,
        load_tokenizer,
    )


def __getattr__(name: str):
    if name in {"load_base_model", "attach_adapter", "attach_lora", "load_tokenizer"}:
        from lora_finetune.models import peft_model

        return getattr(peft_model, name)
    if name == "merge_and_save":
        from lora_finetune.models.merge import merge_and_save

        return merge_and_save
    raise AttributeError(name)
