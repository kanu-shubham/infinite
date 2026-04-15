"""
Fine-tune CodeT5 on the React Native codegen dataset.

Usage
-----
# 1. Preprocess the data first (only needed once)
python data/preprocess.py

# 2. Run fine-tuning
python training/train.py

# 3. (Optional) Monitor with TensorBoard
tensorboard --logdir checkpoints/runs

The script uses HuggingFace Seq2SeqTrainer which handles:
  - gradient accumulation
  - mixed-precision (fp16)
  - learning-rate scheduling with warm-up
  - best-checkpoint saving (by validation loss)
  - TensorBoard logging
"""

import os
import sys
import logging

# Make sure imports resolve from the project root when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)

from training.config import cfg
from training.dataset import RNCodeDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── 1. Reproducibility ────────────────────────────────────────────────────────

set_seed(cfg.seed)


# ── 2. Tokenizer & Model ──────────────────────────────────────────────────────

logger.info(f"Loading tokenizer and model: {cfg.model_name}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

logger.info(
    f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
)


# ── 3. Datasets ───────────────────────────────────────────────────────────────

logger.info("Loading datasets …")
train_dataset = RNCodeDataset(
    cfg.train_file, tokenizer, cfg.max_input_length, cfg.max_target_length
)
val_dataset = RNCodeDataset(
    cfg.val_file, tokenizer, cfg.max_input_length, cfg.max_target_length
)
logger.info(f"  Train: {len(train_dataset)} examples")
logger.info(f"  Val:   {len(val_dataset)} examples")


# ── 4. Data Collator ──────────────────────────────────────────────────────────

# Dynamically pads each batch to the longest sequence in that batch,
# which is more efficient than static padding to max_length.
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if cfg.fp16 else None,
)


# ── 5. Training Arguments ─────────────────────────────────────────────────────

total_steps = (
    len(train_dataset)
    // (cfg.batch_size * cfg.gradient_accumulation_steps)
    * cfg.num_epochs
)
warmup_steps = int(total_steps * cfg.warmup_ratio)

training_args = Seq2SeqTrainingArguments(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epochs,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    warmup_steps=warmup_steps,
    lr_scheduler_type=cfg.lr_scheduler,
    fp16=cfg.fp16 and torch.cuda.is_available(),
    evaluation_strategy="steps",
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.eval_steps,
    save_total_limit=cfg.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,   # needed for generation-based metrics
    logging_steps=cfg.logging_steps,
    report_to=cfg.report_to,
    seed=cfg.seed,
)

logger.info(f"Total optimiser steps: {total_steps}  |  Warmup: {warmup_steps}")


# ── 6. Trainer ────────────────────────────────────────────────────────────────

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


# ── 7. Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting fine-tuning …")
    train_result = trainer.train()

    # Save the final best model and tokenizer
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Training complete. Best model saved to: {cfg.output_dir}")
    logger.info(
        "Next step: python evaluation/evaluate.py  "
        "to get BLEU scores on the test set."
    )
