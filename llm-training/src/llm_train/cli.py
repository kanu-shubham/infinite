"""Unified CLI: ``llm-train {sft,dpo,distill,eval,serve,merge-lora} --config ...``"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from llm_train.utils.logging import setup_logging

app = typer.Typer(help="Production-grade LLM post-training CLI", add_completion=False)


@app.callback()
def _init() -> None:
    setup_logging()


@app.command()
def sft(config: Annotated[Path, typer.Option("--config", "-c", exists=True)]) -> None:
    """Run an SFT training job."""
    from llm_train.config import SFTConfig, load_config
    from llm_train.training.sft import run_sft

    cfg = load_config(config)
    if not isinstance(cfg, SFTConfig):
        raise typer.BadParameter(f"Expected task=sft, got task={cfg.task}")
    run_sft(cfg, source_config_path=str(config))


@app.command()
def dpo(config: Annotated[Path, typer.Option("--config", "-c", exists=True)]) -> None:
    """Run a DPO training job."""
    from llm_train.config import DPOConfig, load_config
    from llm_train.training.dpo import run_dpo

    cfg = load_config(config)
    if not isinstance(cfg, DPOConfig):
        raise typer.BadParameter(f"Expected task=dpo, got task={cfg.task}")
    run_dpo(cfg, source_config_path=str(config))


@app.command()
def distill(config: Annotated[Path, typer.Option("--config", "-c", exists=True)]) -> None:
    """Run a knowledge distillation job."""
    from llm_train.config import DistillConfig, load_config
    from llm_train.training.distill import run_distill

    cfg = load_config(config)
    if not isinstance(cfg, DistillConfig):
        raise typer.BadParameter(f"Expected task=distill, got task={cfg.task}")
    run_distill(cfg, source_config_path=str(config))


@app.command()
def evaluate(
    model: Annotated[str, typer.Option("--model", "-m")],
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/eval.jsonl"),
    batch_size: int = 8,
) -> None:
    """Run the eval harness against a checkpoint."""
    from llm_train.evaluation.harness import run_eval

    run_eval(str(model), str(dataset), str(output), batch_size=batch_size)


@app.command(name="merge-lora")
def merge_lora(
    adapter: Annotated[Path, typer.Option(exists=True)],
    base: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Merge a LoRA adapter into its base model and save the result."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output))
    tok.save_pretrained(str(output))
    typer.echo(f"Merged model written to {output}")


@app.command()
def serve(
    model: Annotated[str, typer.Option("--model", "-m")],
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Launch the FastAPI inference server."""
    import os

    import uvicorn

    os.environ["LLM_TRAIN_MODEL_PATH"] = model
    uvicorn.run("llm_train.inference.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
