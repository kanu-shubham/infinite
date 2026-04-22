from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_train.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sft" in result.stdout
    assert "dpo" in result.stdout
    assert "distill" in result.stdout


def test_cli_sft_rejects_dpo_config(configs_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sft", "--config", str(configs_dir / "dpo" / "qwen_0_5b.yaml")])
    # BadParameter exit is 2, but we only check it's non-zero because Typer
    # wraps the error path.
    assert result.exit_code != 0


def test_cli_distill_rejects_sft_config(configs_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["distill", "--config", str(configs_dir / "sft" / "qwen_0_5b.yaml")]
    )
    assert result.exit_code != 0
