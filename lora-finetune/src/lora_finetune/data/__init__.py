"""Data loading, templating, and tokenization.

Submodules are imported lazily so light-weight consumers (e.g. template users
in tests) don't pay for `datasets`/`torch` imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["build_datasets", "format_example", "get_template"]

if TYPE_CHECKING:  # pragma: no cover
    from lora_finetune.data.dataset import build_datasets
    from lora_finetune.data.templates import format_example, get_template


def __getattr__(name: str):
    if name in {"format_example", "get_template"}:
        from lora_finetune.data import templates

        return getattr(templates, name)
    if name == "build_datasets":
        from lora_finetune.data.dataset import build_datasets

        return build_datasets
    raise AttributeError(name)
