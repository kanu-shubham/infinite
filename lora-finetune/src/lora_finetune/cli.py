"""Typer-based CLI: train / evaluate / merge / serve."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from lora_finetune.config import load_config
from lora_finetune.logging_utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="LoRA/QLoRA fine-tuning CLI")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    """Run LoRA/QLoRA training for an experiment config."""
    setup_logging()
    from lora_finetune.training import run_training

    cfg = load_config(config)
    metrics = run_training(cfg)
    rprint({"metrics": metrics})


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    adapter: Path = typer.Option(..., "--adapter", "-a", exists=True),
    output: Path = typer.Option(Path("outputs/eval.json"), "--output", "-o"),
) -> None:
    """Evaluate a trained adapter against the eval split."""
    setup_logging()
    from lora_finetune.evaluation import evaluate_adapter

    cfg = load_config(config)
    results = evaluate_adapter(cfg, adapter, output)
    rprint(results)


@app.command()
def merge(
    adapter: Path = typer.Option(..., "--adapter", "-a", exists=True),
    output: Path = typer.Option(..., "--output", "-o"),
    base: str | None = typer.Option(None, "--base"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Merge adapter weights into the base model and save to disk."""
    setup_logging()
    from lora_finetune.models.merge import merge_and_save

    path = merge_and_save(adapter, output, base, dtype, trust_remote_code)
    rprint(f"[green]Merged model saved to {path}[/green]")


@app.command()
def serve(
    model: Path = typer.Option(..., "--model", "-m", exists=True),
    adapter: Path | None = typer.Option(None, "--adapter"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    from lora_finetune.serving.api import create_app

    setup_logging()
    application = create_app(str(model), str(adapter) if adapter else None, max_concurrency)
    uvicorn.run(application, host=host, port=port, log_config=None)


@app.command()
def validate_config(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    """Validate a YAML config file without running anything."""
    cfg = load_config(config)
    rprint(f"[green]OK[/green] experiment={cfg.experiment} qlora={cfg.is_qlora}")


if __name__ == "__main__":
    app()
