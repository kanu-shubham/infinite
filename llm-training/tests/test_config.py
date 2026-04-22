from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_train.config import (
    DPOConfig,
    DistillConfig,
    SFTConfig,
    dump_config,
    load_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_sft_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "sft" / "qwen_0_5b.yaml")
    assert isinstance(cfg, SFTConfig)
    assert cfg.task == "sft"
    assert cfg.lora.enabled
    assert cfg.completion_only_loss is True


def test_load_dpo_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "dpo" / "qwen_0_5b.yaml")
    assert isinstance(cfg, DPOConfig)
    assert cfg.task == "dpo"
    assert 0.0 < cfg.beta <= 1.0
    assert cfg.loss_type == "sigmoid"


def test_load_distill_config(configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "distill" / "qwen_7b_to_0_5b.yaml")
    assert isinstance(cfg, DistillConfig)
    assert cfg.task == "distill"
    assert cfg.alpha_ce + cfg.alpha_kd > 0
    assert cfg.mode in {"logit", "sequence"}


def test_missing_task_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "bad.yaml", {"model": {"name_or_path": "x"}})
    with pytest.raises(ValueError, match="task"):
        load_config(path)


def test_unknown_task_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "bad.yaml", {"task": "nonsense"})
    with pytest.raises(Exception):  # pydantic ValidationError
        load_config(path)


def test_distill_rejects_both_quant_flags(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "bad.yaml",
        {
            "task": "distill",
            "student": {"name_or_path": "a", "load_in_4bit": True, "load_in_8bit": True},
            "teacher": {"name_or_path": "b"},
            "data": {"train_path": "x.jsonl"},
        },
    )
    with pytest.raises(Exception):
        load_config(path)


def test_dump_and_reload_roundtrip(tmp_path: Path, configs_dir: Path) -> None:
    cfg = load_config(configs_dir / "sft" / "qwen_0_5b.yaml")
    out = tmp_path / "resolved.yaml"
    dump_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.model_dump() == cfg.model_dump()
