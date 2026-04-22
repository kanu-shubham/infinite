from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_train.config import DataConfig
from llm_train.data.loading import load_raw_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "train.jsonl"
    _write_jsonl(p, [{"prompt": "hi", "response": "hello"} for _ in range(5)])
    cfg = DataConfig(train_path=str(p), format="jsonl", shuffle=False)
    ds = load_raw_dataset(cfg, "train")
    assert len(ds) == 5
    assert ds[0]["prompt"] == "hi"


def test_max_samples_truncates(tmp_path: Path) -> None:
    p = tmp_path / "train.jsonl"
    _write_jsonl(p, [{"prompt": str(i), "response": str(i)} for i in range(10)])
    cfg = DataConfig(train_path=str(p), format="jsonl", shuffle=False, max_samples=3)
    ds = load_raw_dataset(cfg, "train")
    assert len(ds) == 3


def test_missing_file_raises(tmp_path: Path) -> None:
    cfg = DataConfig(train_path=str(tmp_path / "nope.jsonl"), format="jsonl")
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(cfg, "train")


def test_invalid_split_raises(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [{"prompt": "a", "response": "b"}])
    cfg = DataConfig(train_path=str(p), format="jsonl")
    with pytest.raises(ValueError):
        load_raw_dataset(cfg, "not-a-split")  # type: ignore[arg-type]


def test_sft_sample_jsonl_is_valid(examples_dir: Path) -> None:
    rows = [
        json.loads(line)
        for line in (examples_dir / "sft_sample.jsonl").read_text().splitlines()
        if line
    ]
    assert len(rows) >= 1
    for r in rows:
        assert "messages" in r or ("prompt" in r and "response" in r)


def test_dpo_sample_jsonl_is_valid(examples_dir: Path) -> None:
    rows = [
        json.loads(line)
        for line in (examples_dir / "dpo_sample.jsonl").read_text().splitlines()
        if line
    ]
    assert len(rows) >= 1
    for r in rows:
        assert {"prompt", "chosen", "rejected"} <= set(r)
