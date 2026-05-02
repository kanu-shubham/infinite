"""Post-training **AWQ** (Activation-aware Weight Quantization) of a merged model.

AWQ identifies the *salient* weight channels (top ~1% by activation magnitude)
and protects them from quantization, giving better INT4 quality than naive
round-to-nearest. Inference is fast because dequantization fuses with the
matmul.

Usage:
    python scripts/quantize_awq.py \\
        --model outputs/merged \\
        --output outputs/merged-awq-4bit

Requires ``autoawq>=0.2.6`` and a CUDA GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--zero-point", action="store_true", default=True)
    p.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"])
    args = p.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, device_map="auto")

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version,
    }

    print(f"[awq] Quantizing {args.model} -> {args.bits}-bit, gs={args.group_size}")
    model.quantize(tokenizer, quant_config=quant_config)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[awq] Wrote {args.output}")


if __name__ == "__main__":
    main()
