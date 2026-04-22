from __future__ import annotations

from pathlib import Path

from llm_train.config import load_config
from llm_train.training.common import build_hf_training_args, snapshot_run


def test_build_hf_training_args(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "sft" / "qwen_0_5b.yaml")
    args = build_hf_training_args(cfg.training, cfg.optimizer)
    assert args.per_device_train_batch_size == cfg.training.per_device_train_batch_size
    assert args.learning_rate == cfg.optimizer.learning_rate
    assert args.bf16 == cfg.training.bf16
    assert args.gradient_checkpointing == cfg.training.gradient_checkpointing


def test_snapshot_writes_manifest(configs_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(configs_dir / "sft" / "qwen_0_5b.yaml")
    out = snapshot_run(cfg, str(tmp_path / "run"), source_config_path=None)
    assert (out / "resolved_config.yaml").exists()
    assert (out / "run_meta.json").exists()
