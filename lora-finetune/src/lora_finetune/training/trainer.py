"""Training entry point for LoRA and QLoRA fine-tuning.

Wraps HuggingFace Trainer with the project's config, PEFT adapter attach,
custom collator, early stopping, and MLflow/W&B logging.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import transformers
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from lora_finetune.config import ExperimentConfig
from lora_finetune.data.collator import CausalCollator
from lora_finetune.data.dataset import build_datasets
from lora_finetune.logging_utils import get_logger
from lora_finetune.models.peft_model import attach_lora, load_base_model, load_tokenizer

logger = get_logger(__name__)


def _to_training_args(cfg: ExperimentConfig) -> TrainingArguments:
    t = cfg.training
    out = Path(t.output_dir) / cfg.experiment
    out.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(out),
        run_name=t.run_name or cfg.experiment,
        seed=t.seed,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim=t.optim,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        bf16=t.bf16,
        fp16=t.fp16,
        tf32=t.tf32,
        report_to=t.report_to,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=t.neftune_noise_alpha,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def run_training(cfg: ExperimentConfig) -> dict[str, float]:
    set_seed(cfg.training.seed)
    transformers.logging.set_verbosity_info()

    tokenizer = load_tokenizer(cfg)
    train_ds, eval_ds = build_datasets(cfg.data, tokenizer)
    base_model = load_base_model(cfg)
    model = attach_lora(base_model, cfg)

    args = _to_training_args(cfg)
    collator = CausalCollator(tokenizer=tokenizer)

    callbacks = []
    if cfg.training.early_stopping_patience and eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(cfg.training.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Starting training", extra={"experiment": cfg.experiment})
    train_result = trainer.train(resume_from_checkpoint=_find_resume(args.output_dir))
    metrics = dict(train_result.metrics)

    adapter_dir = Path(args.output_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)
        if "eval_loss" in eval_metrics:
            metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])

    (Path(args.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Training complete", extra={"adapter_dir": str(adapter_dir)})
    return metrics


def _find_resume(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [p for p in os.listdir(output_dir) if p.startswith("checkpoint-")]
    return output_dir if ckpts else None
