"""DPO training entrypoint.

Uses TRL's ``DPOTrainer``. When LoRA is enabled the reference model is
implicit (TRL disables the adapter to recover the base policy), otherwise
we load an explicit frozen copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

from llm_train.data.dpo import build_dpo_datasets
from llm_train.models.loader import apply_lora, load_model_and_tokenizer
from llm_train.training.common import build_hf_training_args, snapshot_run
from llm_train.utils.logging import get_logger
from llm_train.utils.seed import set_global_seed

if TYPE_CHECKING:
    from llm_train.config import DPOConfig

log = get_logger(__name__)


def _to_trl_dpo_args(cfg: DPOConfig) -> TRLDPOConfig:
    base = build_hf_training_args(cfg.training, cfg.optimizer)
    # DPOConfig is a dataclass that extends TrainingArguments; round-trip
    # through its dict to pick up DPO-specific fields.
    base_dict = base.to_dict()
    base_dict.update(
        {
            "beta": cfg.beta,
            "loss_type": cfg.loss_type,
            "label_smoothing": cfg.label_smoothing,
            "max_prompt_length": cfg.max_prompt_length,
            "max_length": cfg.max_length,
        }
    )
    # TRLDPOConfig has fields TrainingArguments does not recognize — filter.
    valid_fields = {f.name for f in TRLDPOConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in base_dict.items() if k in valid_fields}
    return TRLDPOConfig(**filtered)


def run_dpo(cfg: DPOConfig, *, source_config_path: str | None = None) -> str:
    set_global_seed(cfg.training.seed)
    out = snapshot_run(cfg, cfg.training.output_dir, source_config_path=source_config_path)

    model, tokenizer = load_model_and_tokenizer(cfg.model, cfg.tokenizer, for_training=True)
    model = apply_lora(model, cfg.lora)

    ref_model = None
    if cfg.ref_model is not None and not cfg.lora.enabled:
        ref_model, _ = load_model_and_tokenizer(cfg.ref_model, cfg.tokenizer, for_training=False)
        for p in ref_model.parameters():
            p.requires_grad_(False)

    train_ds, eval_ds = build_dpo_datasets(cfg.data, tokenizer)

    trl_args = _to_trl_dpo_args(cfg)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trl_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    log.info(
        "Starting DPO: beta=%.3f loss=%s train=%d eval=%s",
        cfg.beta, cfg.loss_type, len(train_ds), bool(eval_ds),
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    if eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    log.info("DPO complete → %s", out)
    return str(out)
