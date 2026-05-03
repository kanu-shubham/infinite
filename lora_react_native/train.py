"""QLoRA fine-tune of a coding LLM on the React Native SFT dataset.

Run:
    python prepare_data.py
    accelerate launch train.py        # multi-GPU
    # or:
    python train.py                   # single GPU

Outputs a PEFT adapter at CFG.output_dir. Merge with merge_and_export.py if
you want a standalone HF model directory.
"""
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from config import CFG


def build_model_and_tokenizer():
    bnb_config = None
    if CFG.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=CFG.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, CFG.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # Coding models often ship without an explicit pad token.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # required when grad checkpointing is on
    if CFG.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CFG.lora_target_modules,
    )


def main() -> None:
    assert os.path.exists(CFG.train_path), (
        f"{CFG.train_path} not found — run prepare_data.py first."
    )

    train_ds = load_dataset("json", data_files=CFG.train_path, split="train")
    eval_ds = load_dataset("json", data_files=CFG.eval_path, split="train")

    model, tokenizer = build_model_and_tokenizer()
    peft_config = build_lora_config()

    sft_config = SFTConfig(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.per_device_batch_size,
        per_device_eval_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=CFG.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=CFG.warmup_ratio,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_seq_length=CFG.max_seq_len,
        packing=False,  # keep messages independent for clean loss masking
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    print(f"[train] adapter saved to {CFG.output_dir}")


if __name__ == "__main__":
    main()
