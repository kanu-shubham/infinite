from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["create_app"]

if TYPE_CHECKING:  # pragma: no cover
    from lora_finetune.serving.api import create_app


def __getattr__(name: str):
    if name == "create_app":
        from lora_finetune.serving.api import create_app

        return create_app
    raise AttributeError(name)
