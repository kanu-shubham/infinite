"""Central config for the React Native LoRA pipeline."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Base coding model. Swap for any HF causal LM you have access to.
    # Qwen2.5-Coder is a strong open coding model; 7B fits in 24GB with QLoRA.
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Where prepared data + adapter weights are written.
    data_dir: str = "data"
    output_dir: str = "outputs/rn-coder-lora"

    # Raw dataset to seed from. Replace with your own JSONL of
    # {"instruction": ..., "code": ...} pairs scraped from RN repos.
    seed_dataset_path: str = "data/rn_seed.jsonl"
    train_path: str = "data/train.jsonl"
    eval_path: str = "data/eval.jsonl"

    # Training hyperparameters.
    max_seq_len: int = 2048
    lr: float = 2e-4
    epochs: int = 3
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.03
    eval_ratio: float = 0.05  # fraction held out for eval

    # LoRA hyperparameters. r=16 is a good default for code; bump to 32 for
    # larger domain shift (e.g. proprietary component libraries).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target every linear projection in the attention + MLP blocks. This is
    # the "QLoRA paper" recipe and matters more than r for code quality.
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # 4-bit quantization (QLoRA) keeps a 7B trainable on a single 24GB GPU.
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


CFG = Config()
