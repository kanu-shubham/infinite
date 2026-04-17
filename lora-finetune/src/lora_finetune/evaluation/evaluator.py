"""Evaluate a trained LoRA adapter against an eval dataset.

Computes loss/perplexity plus optional generation metrics (ROUGE, BLEU,
exact-match) on a held-out split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from lora_finetune.config import ExperimentConfig
from lora_finetune.data.dataset import build_datasets
from lora_finetune.data.templates import format_example
from lora_finetune.evaluation.metrics import bleu_score, exact_match, perplexity_from_loss, rouge_scores
from lora_finetune.logging_utils import get_logger
from lora_finetune.models.peft_model import load_base_model, load_tokenizer

logger = get_logger(__name__)


@torch.no_grad()
def _loss_over_dataset(model, tokenizer, eval_ds, batch_size: int) -> float:
    from lora_finetune.data.collator import CausalCollator
    from torch.utils.data import DataLoader

    model.eval()
    collator = CausalCollator(tokenizer=tokenizer)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)

    from lora_finetune.data.constants import IGNORE_INDEX

    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
        total_loss += float(out.loss) * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _generate_predictions(
    model, tokenizer, eval_ds, cfg: ExperimentConfig
) -> tuple[list[str], list[str]]:
    model.eval()
    preds, refs = [], []
    limit = cfg.evaluation.num_eval_samples or len(eval_ds)
    text_col = cfg.data.response_field or cfg.data.text_field

    for i, example in enumerate(eval_ds.select(range(min(limit, len(eval_ds))))):
        prompt = example.get(cfg.data.prompt_field, "") if cfg.data.prompt_field else ""
        prompt_text = format_example(cfg.data.template, prompt, None, cfg.data.system_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(text)
        refs.append(str(example.get(text_col, "")))
    return preds, refs


def evaluate_adapter(
    cfg: ExperimentConfig,
    adapter_dir: str | Path,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(cfg)
    _, eval_ds = build_datasets(cfg.data, tokenizer)
    if eval_ds is None:
        raise ValueError("No eval dataset available (check eval_file or eval_split_ratio).")

    base = load_base_model(cfg)
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    results: dict[str, Any] = {}
    if "loss" in cfg.evaluation.metrics or "perplexity" in cfg.evaluation.metrics:
        loss = _loss_over_dataset(model, tokenizer, eval_ds, cfg.training.per_device_eval_batch_size)
        results["loss"] = loss
        results["perplexity"] = perplexity_from_loss(loss)

    gen_metrics = {"rouge", "bleu", "exact_match"} & set(cfg.evaluation.metrics)
    if gen_metrics:
        preds, refs = _generate_predictions(model, tokenizer, eval_ds, cfg)
        if "exact_match" in gen_metrics:
            results["exact_match"] = exact_match(preds, refs)
        if "rouge" in gen_metrics:
            results["rouge"] = rouge_scores(preds, refs)
        if "bleu" in gen_metrics:
            results["bleu"] = bleu_score(preds, refs)

    logger.info("Evaluation complete", extra={"results": results})
    if output_file:
        Path(output_file).write_text(json.dumps(results, indent=2))
    return results
