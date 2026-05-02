"""Merge a trained LoRA adapter into the base model weights.

Merged weights can be served directly (no PEFT runtime) and are suitable for
vLLM or TGI. Merging a QLoRA adapter requires reloading the base model in
full precision first.
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_finetune.logging_utils import get_logger

logger = get_logger(__name__)


def merge_and_save(
    adapter_dir: str | Path,
    output_dir: str | Path,
    base_model_override: str | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Path:
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype
    ]

    peft_cfg_path = adapter_dir / "adapter_config.json"
    if not peft_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found at {peft_cfg_path}")

    import json

    base = base_model_override or json.loads(peft_cfg_path.read_text())["base_model_name_or_path"]
    logger.info("Merging adapter", extra={"base": base, "adapter": str(adapter_dir)})

    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)

    logger.info("Saved merged model", extra={"output_dir": str(output_dir)})
    return output_dir
