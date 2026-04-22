"""Inference utilities shared by the eval harness and the serving layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


def load_for_inference(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a merged / checkpoint model + tokenizer for inference."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # important for batched decoding
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Iterable[list[dict[str, str]] | str],
    params: GenerationParams | None = None,
) -> list[str]:
    """Generate completions for a batch of chat-style or raw prompts."""
    params = params or GenerationParams()

    rendered: list[str] = []
    for p in prompts:
        if isinstance(p, list):
            rendered.append(
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            )
        else:
            rendered.append(p)

    enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    out = model.generate(
        **enc,
        max_new_tokens=params.max_new_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        do_sample=params.do_sample,
        repetition_penalty=params.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Strip the prompt — generate returns prompt+completion.
    completions: list[str] = []
    for i, seq in enumerate(out):
        prompt_len = int(enc["attention_mask"][i].sum())
        completions.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    return completions
