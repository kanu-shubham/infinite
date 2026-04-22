#!/usr/bin/env python
"""Merge a LoRA adapter into its base model.

Produces a standard HF-format directory that can be loaded with
``AutoModelForCausalLM.from_pretrained`` — no PEFT dependency needed
at inference time.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"Merged model written to {args.output}")


if __name__ == "__main__":
    main()
