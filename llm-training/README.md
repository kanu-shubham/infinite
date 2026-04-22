# llm-train

Production-grade LLM post-training covering:

- **SFT** — supervised fine-tuning (chat-template-aware, completion-only loss, LoRA/QLoRA)
- **DPO** — direct preference optimization on `(prompt, chosen, rejected)` pairs (via TRL)
- **Distillation** — logit-level knowledge distillation with a frozen teacher, plus sequence-level distillation via a teacher-data-generation script

Built on `transformers`, `trl`, `peft`, `accelerate`, `datasets`.

## Why this exists

Most training "examples" on the web are notebooks. This project is the
opinionated, minimal scaffolding you actually need to put a post-training
pipeline into production: typed configs, reproducible run artifacts,
distributed launchers, Docker, CI, a test suite, and an inference server.

## Quickstart

```bash
cd llm-training
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. SFT a small model on the sample data
llm-train sft --config configs/sft/qwen_0_5b.yaml

# 2. DPO on top of the SFT checkpoint
llm-train dpo --config configs/dpo/qwen_0_5b.yaml

# 3. Distill a 7B teacher into a 0.5B student
llm-train distill --config configs/distill/qwen_7b_to_0_5b.yaml

# 4. Serve the final model
llm-train serve --model outputs/dpo_qwen_0_5b --port 8000
```

For multi-GPU:

```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_sft.py --config configs/sft/qwen_0_5b.yaml
```

## Layout

```
llm-training/
├── src/llm_train/
│   ├── config.py              # Pydantic config schemas (task-discriminated)
│   ├── data/                  # SFT / DPO / distill dataset builders
│   ├── models/                # Model + tokenizer + LoRA loading
│   ├── training/              # SFT, DPO, and custom distillation trainers
│   ├── evaluation/            # Generate + lightweight eval harness
│   ├── inference/             # FastAPI OpenAI-compatible server
│   ├── utils/                 # Logging, seeding
│   └── cli.py                 # `llm-train` Typer CLI
├── scripts/                   # accelerate-compatible entrypoints
├── configs/                   # Example YAML configs per task
├── accelerate_configs/        # single-GPU, DeepSpeed ZeRO-3, FSDP
├── data/examples/             # Toy SFT + DPO data to smoke-test the pipeline
├── tests/                     # pytest suite
├── docs/                      # Pipeline-specific docs
├── docker/                    # GPU Dockerfile
└── Makefile                   # `make sft | dpo | distill | serve | test`
```

## Pipelines

| Use case                                  | Pipeline | Config                                     |
| ----------------------------------------- | -------- | ------------------------------------------ |
| Teach style / instruction following       | SFT      | `configs/sft/qwen_0_5b.yaml`               |
| Align to human preferences                | DPO      | `configs/dpo/qwen_0_5b.yaml`               |
| Compress a big model into a small student | Distill  | `configs/distill/qwen_7b_to_0_5b.yaml`     |

Full details in [`docs/`](docs/).

## Configuration model

Every run is a single YAML file with a discriminating `task:` field:

```yaml
task: sft | dpo | distill
```

Configs are parsed into typed Pydantic models (`SFTConfig` / `DPOConfig` /
`DistillConfig`). At the start of every run the resolved config is dumped
to `<output_dir>/resolved_config.yaml` and the source file is copied to
`source_config.yaml` — so a checkpoint is always reproducible without
the rest of the repo.

## Hardware notes

| Scenario                           | Recommended                         |
| ---------------------------------- | ----------------------------------- |
| Single 24 GB GPU, 0.5B–3B models   | `accelerate_configs/single_gpu.yaml` |
| 8× A100/H100                        | `deepspeed_zero3.yaml` or `fsdp.yaml` |
| 7B model on a 24 GB GPU             | Enable `load_in_4bit` + LoRA (QLoRA) |

## Testing

```bash
make test-fast      # config, data, CLI, loss-math tests
make lint
make typecheck
```

No GPU, no network, no model downloads required for the fast suite.

## Production considerations

- **Reproducibility** — every run snapshots its config + seeds every RNG.
- **Resumption** — set `training.resume_from_checkpoint` to any HF checkpoint dir.
- **Observability** — W&B by default; `report_to: []` disables it.
- **Structured logging** — set `LOG_FORMAT=json` for log shippers.
- **Security** — the Docker image runs as a non-root user; secrets via `.env`.

## License

Apache-2.0.
