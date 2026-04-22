#!/usr/bin/env python
"""Generate teacher responses for sequence-level distillation.

Reads a JSONL of prompts, generates with a teacher model, and writes a
JSONL usable by the SFT pipeline (``{"messages": [...]}`` rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_train.evaluation.generate import (
    GenerationParams,
    batch_generate,
    load_for_inference,
)
from llm_train.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, help="Teacher model path or HF id")
    parser.add_argument("--prompts", required=True, help="Input JSONL (prompt or messages)")
    parser.add_argument("--output", required=True, help="Output JSONL (messages)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    rows: list[dict] = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line]
    prompts: list = []
    for r in rows:
        if "messages" in r:
            prompts.append(r["messages"])
        elif "prompt" in r:
            prompts.append([{"role": "user", "content": r["prompt"]}])
        else:
            raise ValueError(f"row missing prompt/messages: {r}")

    model, tokenizer = load_for_inference(args.teacher)
    params = GenerationParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w") as f:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            completions = batch_generate(model, tokenizer, batch, params=params)
            for src, completion in zip(batch, completions, strict=True):
                messages = list(src) if isinstance(src, list) else [{"role": "user", "content": src}]
                messages.append({"role": "assistant", "content": completion.strip()})
                f.write(json.dumps({"messages": messages}) + "\n")
                total += 1
            log.info("generated %d/%d", min(i + args.batch_size, len(prompts)), len(prompts))
    log.info("wrote %d teacher rows to %s", total, out_path)


if __name__ == "__main__":
    main()
