"""
Training hyperparameters and paths for RN-CodeGen fine-tuning.

All values here are sensible defaults that run on a free-tier
Colab/Kaggle GPU (T4, ~15 GB VRAM).  For a local A100 you can
increase BATCH_SIZE and drop GRADIENT_ACCUMULATION_STEPS.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    # Salesforce/codet5-small  →  60 M params  (fits on any GPU, even CPU)
    # Salesforce/codet5-base   → 220 M params  (recommended for best quality)
    model_name: str = "Salesforce/codet5-small"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_file: str = str(ROOT / "data" / "processed" / "train.jsonl")
    val_file: str   = str(ROOT / "data" / "processed" / "val.jsonl")
    test_file: str  = str(ROOT / "data" / "processed" / "test.jsonl")

    # Max token lengths.  React Native components rarely exceed 512 tokens.
    max_input_length: int  = 128   # prompt tokens
    max_target_length: int = 512   # generated code tokens

    # ── Training ──────────────────────────────────────────────────────────────
    output_dir: str = str(ROOT / "checkpoints")
    num_epochs: int = 10
    # Effective batch size = batch_size * gradient_accumulation_steps = 16
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1          # 10 % of total steps for LR warm-up
    lr_scheduler: str = "cosine"
    fp16: bool = True                  # set False if running on CPU
    save_total_limit: int = 3          # keep only the 3 best checkpoints
    eval_steps: int = 50               # evaluate every N optimizer steps
    logging_steps: int = 10

    # ── Generation (used during evaluation) ────────────────────────────────────
    num_beams: int = 4
    early_stopping: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    report_to: list = field(default_factory=lambda: ["tensorboard"])


# Singleton used by train.py and evaluate.py
cfg = TrainingConfig()
