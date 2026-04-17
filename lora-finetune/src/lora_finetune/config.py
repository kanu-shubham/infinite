"""Strongly-typed experiment configuration loaded from YAML.

All training/eval/serving runs take a single config object so experiments are
reproducible and auditable. Environment variables can override any field via
``${VAR:-default}`` interpolation in YAML strings.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    use_cache: bool = False
    gradient_checkpointing: bool = True


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    @model_validator(mode="after")
    def _only_one_bit(self) -> QuantizationConfig:
        if self.enabled and self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Choose either 4-bit or 8-bit, not both.")
        return self


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM"] = "CAUSAL_LM"
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    modules_to_save: list[str] | None = None
    use_rslora: bool = False
    use_dora: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str | None = None
    dataset_config: str | None = None
    train_file: str | None = None
    eval_file: str | None = None
    text_field: str = "text"
    prompt_field: str | None = None
    response_field: str | None = None
    template: Literal["chatml", "llama3", "alpaca", "raw"] = "chatml"
    system_prompt: str | None = None
    max_seq_length: int = 2048
    pack_sequences: bool = False
    eval_split_ratio: float = Field(0.02, ge=0.0, le=0.5)
    num_proc: int = 4
    streaming: bool = False

    @model_validator(mode="after")
    def _needs_source(self) -> DataConfig:
        if not self.dataset_name and not self.train_file:
            raise ValueError("Provide `dataset_name` or `train_file`.")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "outputs"
    run_name: str | None = None
    seed: int = 42
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 200
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    report_to: list[str] = Field(default_factory=lambda: ["mlflow"])
    early_stopping_patience: int | None = 3
    neftune_noise_alpha: float | None = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[Literal["loss", "perplexity", "rouge", "bleu", "exact_match"]] = Field(
        default_factory=lambda: ["loss", "perplexity"]
    )
    max_new_tokens: int = 256
    generation_batch_size: int = 4
    num_eval_samples: int | None = None


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrency: int = 8
    default_max_new_tokens: int = 256
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    request_timeout_s: float = 60.0


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @field_validator("experiment")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", v):
            raise ValueError("experiment must be a slug: [A-Za-z0-9_.-]+")
        return v

    @property
    def is_qlora(self) -> bool:
        return self.quantization.enabled


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(:-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def sub(match: re.Match[str]) -> str:
            var, _, default = match.groups()
            return os.environ.get(var, default if default is not None else "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment config with env interpolation."""
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(_expand_env(raw))
