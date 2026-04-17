from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from lora_finetune.config import ExperimentConfig, load_config


def write_yaml(tmp: Path, data: dict) -> Path:
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


MINIMAL = {
    "experiment": "exp1",
    "model": {"name_or_path": "gpt2"},
    "data": {"train_file": "train.jsonl", "prompt_field": "p", "response_field": "r"},
}


def test_loads_minimal(tmp_path: Path) -> None:
    cfg = load_config(write_yaml(tmp_path, MINIMAL))
    assert cfg.experiment == "exp1"
    assert cfg.lora.r == 16
    assert not cfg.is_qlora


def test_rejects_bad_experiment(tmp_path: Path) -> None:
    bad = {**MINIMAL, "experiment": "has space"}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))


def test_expands_env_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/out")
    data = {
        **MINIMAL,
        "training": {"output_dir": "${OUTPUT_DIR:-default_out}"},
    }
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.training.output_dir == "/tmp/out"


def test_env_default_when_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    data = {**MINIMAL, "training": {"output_dir": "${OUTPUT_DIR:-default_out}"}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.training.output_dir == "default_out"


def test_qlora_flag(tmp_path: Path) -> None:
    data = {**MINIMAL, "quantization": {"enabled": True, "load_in_4bit": True}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.is_qlora


def test_requires_data_source(tmp_path: Path) -> None:
    bad = {"experiment": "e", "model": {"name_or_path": "gpt2"}, "data": {}}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    bad = {**MINIMAL, "nonsense": 1}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))


def test_rejects_both_bit_widths() -> None:
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(
            {
                **MINIMAL,
                "quantization": {"enabled": True, "load_in_4bit": True, "load_in_8bit": True},
            }
        )
