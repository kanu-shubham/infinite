from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from lora_finetune.cli import app

runner = CliRunner()


def test_validate_config_ok(tmp_path: Path) -> None:
    cfg = {
        "experiment": "exp1",
        "model": {"name_or_path": "gpt2"},
        "data": {"train_file": "t.jsonl", "prompt_field": "p", "response_field": "r"},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    result = runner.invoke(app, ["validate-config", "--config", str(p)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
