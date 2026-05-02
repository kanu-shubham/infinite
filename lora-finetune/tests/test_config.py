from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lora_finetune.config import (
    ExperimentConfig,
    FullFinetuneAdapter,
    IA3Adapter,
    LoRAAdapter,
    PrefixTuningAdapter,
    PTuningAdapter,
    load_config,
)


def write_yaml(tmp: Path, data: dict) -> Path:
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


MINIMAL = {
    "experiment": "exp1",
    "model": {"name_or_path": "gpt2"},
    "data": {"train_file": "train.jsonl", "prompt_field": "p", "response_field": "r"},
}


def test_loads_minimal_defaults_to_lora(tmp_path: Path) -> None:
    cfg = load_config(write_yaml(tmp_path, MINIMAL))
    assert cfg.experiment == "exp1"
    assert isinstance(cfg.adapter, LoRAAdapter)
    assert cfg.adapter.r == 16
    assert cfg.is_peft
    assert not cfg.is_qlora


def test_explicit_lora(tmp_path: Path) -> None:
    data = {**MINIMAL, "adapter": {"type": "lora", "r": 8, "alpha": 16}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert isinstance(cfg.adapter, LoRAAdapter)
    assert cfg.adapter.r == 8 and cfg.adapter.alpha == 16


def test_prefix_adapter(tmp_path: Path) -> None:
    data = {**MINIMAL, "adapter": {"type": "prefix", "num_virtual_tokens": 50}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert isinstance(cfg.adapter, PrefixTuningAdapter)
    assert cfg.adapter.num_virtual_tokens == 50


def test_ptuning_adapter(tmp_path: Path) -> None:
    data = {
        **MINIMAL,
        "adapter": {"type": "ptuning", "num_virtual_tokens": 20, "encoder_hidden_size": 256},
    }
    cfg = load_config(write_yaml(tmp_path, data))
    assert isinstance(cfg.adapter, PTuningAdapter)
    assert cfg.adapter.encoder_hidden_size == 256


def test_ia3_adapter(tmp_path: Path) -> None:
    data = {
        **MINIMAL,
        "adapter": {"type": "ia3", "target_modules": ["k_proj", "v_proj"]},
    }
    cfg = load_config(write_yaml(tmp_path, data))
    assert isinstance(cfg.adapter, IA3Adapter)
    assert cfg.adapter.target_modules == ["k_proj", "v_proj"]


def test_full_finetune(tmp_path: Path) -> None:
    data = {**MINIMAL, "adapter": {"type": "full"}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert isinstance(cfg.adapter, FullFinetuneAdapter)
    assert not cfg.is_peft


def test_unknown_adapter_type_rejected(tmp_path: Path) -> None:
    data = {**MINIMAL, "adapter": {"type": "nonsense"}}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, data))


def test_rejects_bad_experiment(tmp_path: Path) -> None:
    bad = {**MINIMAL, "experiment": "has space"}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))


def test_expands_env_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/out")
    data = {**MINIMAL, "training": {"output_dir": "${OUTPUT_DIR:-default_out}"}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.training.output_dir == "/tmp/out"


def test_env_default_when_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    data = {**MINIMAL, "training": {"output_dir": "${OUTPUT_DIR:-default_out}"}}
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.training.output_dir == "default_out"


def test_qlora_only_for_lora_plus_quant(tmp_path: Path) -> None:
    qlora = {**MINIMAL, "quantization": {"enabled": True, "load_in_4bit": True}}
    assert load_config(write_yaml(tmp_path, qlora)).is_qlora is True

    prefix_quant = {
        **MINIMAL,
        "adapter": {"type": "prefix"},
        "quantization": {"enabled": True, "load_in_4bit": True},
    }
    # Quant + non-LoRA adapter is *not* QLoRA in the strict sense.
    assert load_config(write_yaml(tmp_path, prefix_quant)).is_qlora is False


def test_requires_data_source(tmp_path: Path) -> None:
    bad = {"experiment": "e", "model": {"name_or_path": "gpt2"}, "data": {}}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))


def test_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
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


def test_serving_adapters_parsed(tmp_path: Path) -> None:
    data = {
        **MINIMAL,
        "serving": {
            "adapters": [
                {"name": "support", "path": "outputs/support", "default": True},
                {"name": "sales", "path": "outputs/sales"},
            ]
        },
    }
    cfg = load_config(write_yaml(tmp_path, data))
    assert len(cfg.serving.adapters) == 2
    assert cfg.serving.adapters[0].default
    assert cfg.serving.adapters[1].name == "sales"


def test_lora_rank_bounds(tmp_path: Path) -> None:
    bad = {**MINIMAL, "adapter": {"type": "lora", "r": 0}}
    with pytest.raises(Exception):
        load_config(write_yaml(tmp_path, bad))
