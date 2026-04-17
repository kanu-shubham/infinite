from __future__ import annotations

import pytest

from lora_finetune.data.templates import format_example, get_template


def test_alpaca_contains_response() -> None:
    text = format_example("alpaca", "Say hi", "hello")
    assert "### Response:" in text
    assert "hello" in text


def test_chatml_user_and_assistant_tags() -> None:
    text = format_example("chatml", "p", "r", system="sys")
    assert "<|im_start|>system" in text
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text


def test_llama3_special_tokens() -> None:
    text = format_example("llama3", "p", "r", system="sys")
    assert "<|begin_of_text|>" in text
    assert "<|start_header_id|>user<|end_header_id|>" in text


def test_inference_prompt_omits_response() -> None:
    text = format_example("chatml", "p", None)
    # No closing tag right after assistant header when response is None.
    assert text.endswith("<|im_start|>assistant\n")


def test_raw_template_roundtrip() -> None:
    text = format_example("raw", "p", "r")
    assert "p" in text and "r" in text


def test_unknown_template_raises() -> None:
    with pytest.raises(ValueError):
        get_template("nope")
