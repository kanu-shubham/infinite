"""Lightweight eval harness.

Runs generation over a JSONL eval set and reports aggregate stats.
This is *not* a replacement for ``lm-eval-harness`` — it's meant for
smoke-checking a training run with a handful of custom prompts.

Row shape:
    {"messages": [...]}  or  {"prompt": "..."}
    optional: {"reference": "..."} for exact-match / substring match
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_train.evaluation.generate import (
    GenerationParams,
    batch_generate,
    load_for_inference,
)
from llm_train.utils.logging import get_logger

log = get_logger(__name__)


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_prompt(row: dict[str, Any]) -> list[dict[str, str]] | str:
    if "messages" in row:
        return row["messages"]
    if "prompt" in row:
        return [{"role": "user", "content": row["prompt"]}]
    raise ValueError(f"Unrecognized row: {row}")


def run_eval(
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    batch_size: int = 8,
    params: GenerationParams | None = None,
) -> dict[str, float]:
    """Generate for every row and write outputs as JSONL.

    Returns simple aggregate metrics (avg output length, substring-match
    rate when references are provided).
    """
    rows = _load_prompts(Path(dataset_path))
    model, tokenizer = load_for_inference(model_path)

    all_out: list[str] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompts = [_as_prompt(r) for r in chunk]
        all_out.extend(batch_generate(model, tokenizer, prompts, params=params))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    match = 0
    n_ref = 0
    total_len = 0
    with out_file.open("w") as f:
        for row, completion in zip(rows, all_out, strict=True):
            record = {**row, "completion": completion}
            f.write(json.dumps(record) + "\n")
            total_len += len(completion)
            if "reference" in row:
                n_ref += 1
                if row["reference"].strip().lower() in completion.lower():
                    match += 1

    metrics = {
        "num_examples": float(len(rows)),
        "avg_completion_chars": total_len / max(len(rows), 1),
    }
    if n_ref > 0:
        metrics["substring_match_rate"] = match / n_ref
    log.info("Eval metrics: %s", metrics)
    return metrics
