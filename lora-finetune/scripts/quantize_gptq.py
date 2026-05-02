"""Post-training **GPTQ** quantization of a merged model.

GPTQ is a one-shot post-training quantizer that uses small calibration data
to minimize per-layer reconstruction error. The result is INT4/INT3 weights
that load with no extra runtime cost — much faster than bitsandbytes for
inference, with similar quality.

Usage:
    python scripts/quantize_gptq.py \\
        --model outputs/merged \\
        --output outputs/merged-gptq-4bit \\
        --bits 4 \\
        --dataset c4

Requires ``optimum>=1.20``, ``auto-gptq>=0.7`` and a CUDA GPU. Calibration
typically takes 5–30 minutes depending on model size.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _calibration_samples(name: str, n: int) -> list[str]:
    """Pull ``n`` short text samples for GPTQ calibration."""
    if name == "c4":
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        return [next(iter(ds))["text"][:1024] for _ in range(n)]
    if Path(name).exists():
        return Path(name).read_text().splitlines()[:n]
    raise ValueError(f"Unknown dataset spec: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir (merged or base).")
    p.add_argument("--output", required=True, help="Where to write the GPTQ model.")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--dataset", default="c4", help="'c4' or path to a text file.")
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--desc-act", action="store_true", help="Activation-order quant (slower, better).")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = _calibration_samples(args.dataset, args.n_samples)

    quant_cfg = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        dataset=samples,
        tokenizer=tokenizer,
    )

    print(f"[gptq] Quantizing {args.model} -> {args.bits}-bit, gs={args.group_size}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", quantization_config=quant_cfg
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"[gptq] Wrote {args.output}")


if __name__ == "__main__":
    main()
