"""Prompt templates for common instruction-tuning formats.

Templates are pure functions that produce a single text string from a
(system, prompt, response) triple. The response is optional so the same
template can be reused for inference.
"""
from __future__ import annotations

from typing import Callable

Template = Callable[[str | None, str, str | None], str]

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response "
    "that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)


def _alpaca(system: str | None, prompt: str, response: str | None) -> str:
    text = ALPACA_TEMPLATE.format(prompt=prompt, response=response or "")
    if system:
        text = f"{system}\n\n{text}"
    return text


def _chatml(system: str | None, prompt: str, response: str | None) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{response or ''}")
    if response is not None:
        parts[-1] += "<|im_end|>"
    return "\n".join(parts)


def _llama3(system: str | None, prompt: str, response: str | None) -> str:
    parts = ["<|begin_of_text|>"]
    if system:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>")
    tail = f"<|start_header_id|>assistant<|end_header_id|>\n\n{response or ''}"
    if response is not None:
        tail += "<|eot_id|>"
    parts.append(tail)
    return "".join(parts)


def _raw(system: str | None, prompt: str, response: str | None) -> str:
    if system:
        return f"{system}\n{prompt}\n{response or ''}"
    return f"{prompt}\n{response or ''}"


_TEMPLATES: dict[str, Template] = {
    "alpaca": _alpaca,
    "chatml": _chatml,
    "llama3": _llama3,
    "raw": _raw,
}


def get_template(name: str) -> Template:
    try:
        return _TEMPLATES[name]
    except KeyError as err:
        raise ValueError(f"Unknown template {name!r}. Known: {list(_TEMPLATES)}") from err


def format_example(
    template: str,
    prompt: str,
    response: str | None = None,
    system: str | None = None,
) -> str:
    return get_template(template)(system, prompt, response)
