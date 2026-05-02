"""Base-model + tokenizer loading and PEFT adapter attachment.

Supports five training modes via the ``adapter.type`` discriminated union:
  * ``lora``    — Low-Rank Adaptation (with optional DoRA / RSLoRA)
  * ``prefix``  — Prefix Tuning (trainable KV prefixes per layer)
  * ``ptuning`` — P-Tuning v2 (trainable prompt encoder)
  * ``ia3``     — (IA)^3 rescaling vectors
  * ``full``    — full fine-tuning (no PEFT wrapper)

Quantization (bitsandbytes 4-bit / 8-bit) is orthogonal to the choice above —
QLoRA = ``adapter.type == 'lora'`` + ``quantization.enabled == True``.
"""
from __future__ import annotations

from typing import Any

import torch
from peft import (
    IA3Config,
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from lora_finetune.config import (
    AdapterConfig,
    ExperimentConfig,
    FullFinetuneAdapter,
    IA3Adapter,
    LoRAAdapter,
    PrefixTuningAdapter,
    PTuningAdapter,
)
from lora_finetune.logging_utils import get_logger

logger = get_logger(__name__)

_DTYPE = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _bnb_config(cfg: ExperimentConfig) -> BitsAndBytesConfig | None:
    q = cfg.quantization
    if not q.enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        load_in_8bit=q.load_in_8bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_DTYPE[q.bnb_4bit_compute_dtype],
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
    )


def load_tokenizer(cfg: ExperimentConfig) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(cfg: ExperimentConfig) -> PreTrainedModel:
    """Load the base model.

    Picks the right loading path:
      * ``model.quant_format == 'gptq' | 'awq'`` — use the on-disk quant config
        (transformers picks up GPTQConfig / AWQConfig automatically).
      * ``quantization.enabled`` — bitsandbytes runtime 4/8-bit (QLoRA).
      * otherwise — plain bf16/fp16/fp32 weights.
    """
    kwargs: dict[str, Any] = {
        "revision": cfg.model.revision,
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation,
        "torch_dtype": _DTYPE[cfg.model.torch_dtype],
        "use_cache": cfg.model.use_cache,
    }
    bnb = _bnb_config(cfg)
    if bnb is not None:
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
    elif cfg.model.quant_format in {"gptq", "awq"}:
        kwargs["device_map"] = "auto"

    logger.info(
        "Loading base model",
        extra={
            "model": cfg.model.name_or_path,
            "bnb_quantized": bnb is not None,
            "ondisk_quant": cfg.model.quant_format,
        },
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)

    if bnb is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.model.gradient_checkpointing
        )
    elif cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def _build_peft_config(adapter: AdapterConfig):
    if isinstance(adapter, LoRAAdapter):
        return LoraConfig(
            r=adapter.r,
            lora_alpha=adapter.alpha,
            lora_dropout=adapter.dropout,
            bias=adapter.bias,
            task_type=adapter.task_type,
            target_modules=adapter.target_modules,
            modules_to_save=adapter.modules_to_save,
            use_rslora=adapter.use_rslora,
            use_dora=adapter.use_dora,
        )
    if isinstance(adapter, PrefixTuningAdapter):
        return PrefixTuningConfig(
            num_virtual_tokens=adapter.num_virtual_tokens,
            prefix_projection=adapter.prefix_projection,
            encoder_hidden_size=adapter.encoder_hidden_size,
            task_type=adapter.task_type,
        )
    if isinstance(adapter, PTuningAdapter):
        return PromptEncoderConfig(
            num_virtual_tokens=adapter.num_virtual_tokens,
            encoder_hidden_size=adapter.encoder_hidden_size,
            encoder_reparameterization_type=adapter.encoder_reparameterization_type,
            task_type=adapter.task_type,
        )
    if isinstance(adapter, IA3Adapter):
        return IA3Config(
            target_modules=adapter.target_modules,
            feedforward_modules=adapter.feedforward_modules,
            task_type=adapter.task_type,
        )
    raise TypeError(f"Unsupported adapter type: {type(adapter).__name__}")


def attach_adapter(model: PreTrainedModel, cfg: ExperimentConfig) -> PreTrainedModel | PeftModel:
    """Wrap the base model with the configured PEFT adapter (or pass through)."""
    if isinstance(cfg.adapter, FullFinetuneAdapter):
        logger.info(
            "Full fine-tuning",
            extra={"trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        )
        return model

    peft_cfg = _build_peft_config(cfg.adapter)
    peft_model = get_peft_model(model, peft_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "Attached PEFT adapter",
        extra={
            "adapter_type": cfg.adapter.type,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
        },
    )
    return peft_model


# Back-compat alias — older imports still work.
attach_lora = attach_adapter
