from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["evaluate_adapter"]

if TYPE_CHECKING:  # pragma: no cover
    from lora_finetune.evaluation.evaluator import evaluate_adapter


def __getattr__(name: str):
    if name == "evaluate_adapter":
        from lora_finetune.evaluation.evaluator import evaluate_adapter

        return evaluate_adapter
    raise AttributeError(name)
