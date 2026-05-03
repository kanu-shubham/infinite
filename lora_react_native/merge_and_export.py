"""Merge the LoRA adapter into the base weights and export a standalone model.

Useful when you want to deploy with vLLM / TGI / llama.cpp without PEFT.
Note: cannot merge directly on top of 4-bit weights — we reload in bf16.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CFG

EXPORT_DIR = f"{CFG.output_dir}-merged"


def main() -> None:
    base = AutoModelForCausalLM.from_pretrained(
        CFG.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, CFG.output_dir).merge_and_unload()
    merged.save_pretrained(EXPORT_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(CFG.output_dir).save_pretrained(EXPORT_DIR)
    print(f"[merge] standalone model written to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
