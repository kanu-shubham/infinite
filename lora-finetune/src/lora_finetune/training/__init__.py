from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["run_training"]

if TYPE_CHECKING:  # pragma: no cover
    from lora_finetune.training.trainer import run_training


def __getattr__(name: str):
    if name == "run_training":
        from lora_finetune.training.trainer import run_training

        return run_training
    raise AttributeError(name)
