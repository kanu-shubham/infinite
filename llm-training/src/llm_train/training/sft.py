"""SFT training entrypoint.

Wraps ``transformers.Trainer`` with:
  - typed config → ``TrainingArguments`` translation
  - LoRA / QLoRA support via PEFT
  - completion-only loss masking (done in the data layer)
  - run artifact snapshot (resolved config + source YAML)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import Trainer

from llm_train.data.sft import build_sft_datasets
from llm_train.models.loader import apply_lora, load_model_and_tokenizer
from llm_train.training.collators import CausalCollator
from llm_train.training.common import build_hf_training_args, snapshot_run
from llm_train.utils.logging import get_logger
from llm_train.utils.seed import set_global_seed

if TYPE_CHECKING:
    from llm_train.config import SFTConfig

log = get_logger(__name__)


def run_sft(cfg: SFTConfig, *, source_config_path: str | None = None) -> str:
    """Run an SFT job and return the final checkpoint directory."""
    set_global_seed(cfg.training.seed)
    out = snapshot_run(cfg, cfg.training.output_dir, source_config_path=source_config_path)

    model, tokenizer = load_model_and_tokenizer(cfg.model, cfg.tokenizer, for_training=True)
    model = apply_lora(model, cfg.lora)

    train_ds, eval_ds = build_sft_datasets(
        cfg.data, cfg.tokenizer, tokenizer,
        completion_only_loss=cfg.completion_only_loss,
    )

    training_args = build_hf_training_args(cfg.training, cfg.optimizer)
    collator = CausalCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    log.info("Starting SFT: %d train rows, eval=%s", len(train_ds), bool(eval_ds))
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    if eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    log.info("SFT complete → %s", out)
    return str(out)
