"""Shared trainer utilities.

This module hides the mapping from our typed ``TrainingConfig`` +
``OptimizerConfig`` onto ``transformers.TrainingArguments`` so each task-
specific trainer stays focused on task logic.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainingArguments

from llm_train.config import dump_config
from llm_train.utils.logging import get_logger

if TYPE_CHECKING:
    from llm_train.config import (
        DPOConfig,
        DistillConfig,
        OptimizerConfig,
        SFTConfig,
        TrainingConfig,
    )

log = get_logger(__name__)


def build_hf_training_args(
    training: TrainingConfig,
    optimizer: OptimizerConfig,
    *,
    extra: dict | None = None,
) -> TrainingArguments:
    """Translate our typed config into ``transformers.TrainingArguments``."""
    kwargs = {
        "output_dir": training.output_dir,
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "gradient_checkpointing": training.gradient_checkpointing,
        "bf16": training.bf16,
        "fp16": training.fp16,
        "logging_steps": training.logging_steps,
        "eval_steps": training.eval_steps,
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "eval_strategy": training.eval_strategy,
        "save_strategy": training.save_strategy,
        "report_to": training.report_to,
        "run_name": training.run_name,
        "seed": training.seed,
        "learning_rate": optimizer.learning_rate,
        "weight_decay": optimizer.weight_decay,
        "warmup_ratio": optimizer.warmup_ratio,
        "lr_scheduler_type": optimizer.lr_scheduler_type,
        "optim": optimizer.optim,
        "max_grad_norm": optimizer.max_grad_norm,
        "adam_beta1": optimizer.adam_beta1,
        "adam_beta2": optimizer.adam_beta2,
        "adam_epsilon": optimizer.adam_epsilon,
        "load_best_model_at_end": False,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    if extra:
        kwargs.update(extra)
    return TrainingArguments(**kwargs)


def snapshot_run(
    cfg: SFTConfig | DPOConfig | DistillConfig,
    output_dir: str,
    source_config_path: str | None = None,
) -> Path:
    """Write a reproducibility manifest to ``output_dir``.

    Includes: resolved config, a copy of the source YAML, and a timestamped
    metadata file. Called at the start of every run so a broken training
    job still leaves enough to diagnose what was attempted.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dump_config(cfg, out / "resolved_config.yaml")
    if source_config_path:
        try:
            shutil.copy2(source_config_path, out / "source_config.yaml")
        except OSError as e:
            log.warning("Could not copy source config: %s", e)

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": cfg.task,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return out
