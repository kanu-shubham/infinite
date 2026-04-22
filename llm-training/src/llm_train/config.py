"""Typed configuration schemas.

Configs are declared as Pydantic models and loaded from YAML. The top-level
``ExperimentConfig`` discriminates on the ``task`` field so a single file can
describe an SFT, DPO, or distillation run.

All paths are resolved relative to the config file's directory unless
absolute, which makes configs portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------------------
# Shared sub-configs
# ---------------------------------------------------------------------------


class ModelConfig(_Base):
    """Model-loading options shared across tasks."""

    name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "bf16"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: str | None = None  # e.g. "auto" for inference; None during training

    @model_validator(mode="after")
    def _exclusive_quant(self) -> ModelConfig:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Set at most one of load_in_4bit / load_in_8bit.")
        return self


class LoraConfig(_Base):
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    modules_to_save: list[str] | None = None


class TokenizerConfig(_Base):
    name_or_path: str | None = None  # defaults to model
    padding_side: Literal["left", "right"] = "right"
    model_max_length: int = 4096
    chat_template: str | None = None  # optional Jinja override


class DataConfig(_Base):
    train_path: str
    eval_path: str | None = None
    # Either a HuggingFace dataset identifier or a local path (jsonl/parquet).
    format: Literal["jsonl", "parquet", "hf"] = "jsonl"
    split_train: str = "train"
    split_eval: str = "validation"
    max_samples: int | None = None
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42


class OptimizerConfig(_Base):
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: Literal["cosine", "linear", "constant", "cosine_with_restarts"] = "cosine"
    optim: str = "adamw_torch"  # e.g. "adamw_torch_fused", "paged_adamw_8bit"
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class TrainingConfig(_Base):
    output_dir: str = "outputs/run"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    report_to: list[str] = Field(default_factory=lambda: ["wandb"])
    run_name: str | None = None
    resume_from_checkpoint: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Task-specific configs
# ---------------------------------------------------------------------------


class SFTConfig(_Base):
    task: Literal["sft"]
    model: ModelConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # SFT-specific
    completion_only_loss: bool = True  # mask the prompt tokens in the loss
    packing: bool = False


class DPOConfig(_Base):
    task: Literal["dpo"]
    model: ModelConfig
    # Reference model is optional; TRL will use a frozen copy of the policy if omitted.
    ref_model: ModelConfig | None = None
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # DPO-specific
    beta: float = 0.1
    loss_type: Literal["sigmoid", "ipo", "hinge", "kto_pair"] = "sigmoid"
    label_smoothing: float = 0.0
    max_prompt_length: int = 1024
    max_length: int = 2048


class DistillConfig(_Base):
    task: Literal["distill"]
    student: ModelConfig
    teacher: ModelConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # Distillation-specific
    mode: Literal["logit", "sequence"] = "logit"
    temperature: float = 2.0
    alpha_ce: float = 0.5  # weight on student CE against hard labels
    alpha_kd: float = 0.5  # weight on KL(student || teacher)
    top_k_logits: int | None = None  # optionally restrict KL to teacher's top-k

    @field_validator("alpha_ce", "alpha_kd")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("alpha weights must be >= 0")
        return v

    @model_validator(mode="after")
    def _weights_sum(self) -> DistillConfig:
        if self.alpha_ce + self.alpha_kd <= 0:
            raise ValueError("alpha_ce + alpha_kd must be positive")
        return self


ExperimentConfig = Annotated[
    SFTConfig | DPOConfig | DistillConfig,
    Field(discriminator="task"),
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ConfigWrapper(_Base):
    """Internal helper to force discriminated-union parsing."""

    cfg: ExperimentConfig


def load_config(path: str | Path) -> SFTConfig | DPOConfig | DistillConfig:
    """Load and validate a YAML config.

    The file must contain a top-level ``task:`` key identifying the pipeline.
    Relative paths inside the config are resolved against the config's
    parent directory at the call sites that need filesystem access.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if "task" not in raw:
        raise ValueError(f"Config {path} is missing required 'task' field.")
    wrapped = ConfigWrapper(cfg=raw)  # type: ignore[arg-type]
    return wrapped.cfg


def dump_config(cfg: SFTConfig | DPOConfig | DistillConfig, path: str | Path) -> None:
    """Dump a resolved config to YAML — used to snapshot every run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
