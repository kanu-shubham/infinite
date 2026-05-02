# lora-finetune

Production-grade fine-tuning and serving of large language models with
**LoRA, QLoRA, DoRA, Prefix-Tuning, P-Tuning v2, (IA)³** and full
fine-tuning. Plus post-training quantization to **GPTQ / AWQ / GGUF** and
**multi-adapter** inference serving.

Built on PyTorch, Transformers, PEFT and bitsandbytes. Ships with an
opinionated config system, a HuggingFace-Trainer pipeline, a FastAPI inference
service, MLflow tracking, Dockerfiles, Kubernetes manifests and an Airflow DAG.

> 📘 **New to fine-tuning?** Read [`docs/concepts.md`](docs/concepts.md)
> first — it covers the math behind LoRA, rank/alpha selection,
> target-module choice, the PEFT/full-FT trade-off, catastrophic forgetting,
> and the quantization landscape (bnb / GPTQ / AWQ / GGUF / mixed precision).

## Features

- **PEFT methods**: LoRA, QLoRA, DoRA, RSLoRA, Prefix-Tuning, P-Tuning v2,
  (IA)³, plus a **full fine-tuning** path — all selected via a single
  `adapter.type` field.
- **Quantization**: bitsandbytes 4-bit NF4 / 8-bit at training time;
  post-training **GPTQ**, **AWQ** and **GGUF** export for fast inference.
- **YAML configs** validated by Pydantic with `${ENV:-default}` interpolation
  and discriminated-union adapter blocks.
- **Training**: HF `Trainer` with gradient checkpointing, paged-AdamW,
  early-stopping, MLflow logging, multi-GPU via `accelerate`.
- **Data**: HF Hub / JSONL / CSV / Parquet, chat/Alpaca/Llama-3/ChatML templates,
  response-only loss masking.
- **Evaluation**: loss, perplexity, ROUGE, BLEU, exact-match.
- **Serving**: FastAPI with **multi-adapter hot-swap**, bearer auth, Prometheus
  metrics, async concurrency bounds, health checks; transparent loading of
  GPTQ/AWQ/bnb-4bit models.
- **Ops**: CPU-only CI tests, two Dockerfiles, docker-compose with MLflow,
  Kubernetes Job + Deployment + HPA, Airflow DAG for weekly re-training.

## Quickstart

```bash
make install-dev
cp .env.example .env           # add HF_TOKEN etc.

# Validate a config
lora-finetune validate-config --config configs/qlora_llama3_8b.yaml

# Train (single GPU)
make train CONFIG=configs/qlora_llama3_8b.yaml

# Train (multi-GPU)
NUM_GPUS=4 scripts/train.sh configs/qlora_llama3_8b.yaml

# Evaluate
make eval CONFIG=configs/qlora_llama3_8b.yaml ADAPTER=outputs/qlora-llama3-8b-instruct/adapter

# Merge adapter into base weights for serving
make merge ADAPTER=outputs/qlora-llama3-8b-instruct/adapter MERGED=outputs/merged

# Serve
make serve MERGED=outputs/merged
```

### Call the API

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain LoRA in one sentence."}
    ],
    "max_new_tokens": 128, "temperature": 0.2
  }'
```

## Project layout

```
lora-finetune/
├── configs/                 YAML experiments
│   ├── qlora_llama3_8b.yaml      QLoRA + LoRA on Llama-3-8B
│   ├── lora_mistral_7b.yaml      Standard LoRA on Mistral-7B
│   ├── prefix_mistral_7b.yaml    Prefix tuning
│   ├── ia3_mistral_7b.yaml       (IA)^3
│   ├── full_ft_small.yaml        Full fine-tuning of TinyLlama
│   └── test_tiny.yaml            CI smoke
├── docs/concepts.md         Math + heuristics + decision tables
├── src/lora_finetune/
│   ├── config.py            Pydantic config (adapter discriminated union)
│   ├── data/                loaders, templates, tokenization, collator
│   ├── models/              base-model loading, PEFT attach, adapter merge
│   ├── training/trainer.py  HF Trainer wrapper
│   ├── evaluation/          loss/perplexity + generation metrics
│   ├── serving/             FastAPI app + schemas (multi-adapter hot-swap)
│   ├── logging_utils.py     structured JSON logging
│   └── cli.py               typer entrypoint: train / eval / merge / serve
├── tests/                   unit tests + fixtures (no GPU required)
├── docker/                  Dockerfile.train & Dockerfile.serve
├── deploy/k8s/              Job, Deployment, Service, HPA
├── deploy/airflow/          weekly training DAG
├── scripts/                 accelerate launcher, GPTQ/AWQ/GGUF exporters
├── Makefile                 common workflows
├── docker-compose.yml       MLflow + train + serve
└── pyproject.toml
```

## Configuration reference

Every run takes a single YAML file mapped onto `ExperimentConfig`. Top-level
sections:

| Section        | Purpose                                                          |
| -------------- | ---------------------------------------------------------------- |
| `model`        | base checkpoint, dtype, attention impl, on-disk quant format     |
| `quantization` | enable QLoRA-style runtime quantization (bnb 4-bit NF4 / 8-bit)  |
| `adapter`      | discriminated union: `lora`, `prefix`, `ptuning`, `ia3`, `full`  |
| `data`         | dataset source, template, max seq length, split                  |
| `training`     | HF `TrainingArguments` surface + early stopping                  |
| `evaluation`   | metrics to compute, generation length                            |
| `serving`      | FastAPI host/port, concurrency cap, multi-adapter mounts         |

### Adapter types

```yaml
adapter:
  type: lora        # plus r, alpha, dropout, target_modules, use_dora, use_rslora
adapter:
  type: prefix      # plus num_virtual_tokens, prefix_projection
adapter:
  type: ptuning     # plus encoder_hidden_size, encoder_reparameterization_type
adapter:
  type: ia3         # plus target_modules, feedforward_modules
adapter:
  type: full        # full fine-tuning, no PEFT wrapper
```

See `configs/qlora_llama3_8b.yaml` for a complete QLoRA example and
`configs/test_tiny.yaml` for the CI smoke config.

## Training pipeline

1. `load_tokenizer` — ensures a pad token and right-padding.
2. `build_datasets` — loads, templates and tokenizes, builds a response-only
   label mask (prompt tokens become `-100`).
3. `load_base_model` — loads the base checkpoint, applies `BitsAndBytesConfig`
   if QLoRA, enables gradient checkpointing.
4. `attach_lora` — wraps with PEFT `LoraConfig` and logs the trainable-parameter
   count.
5. HF `Trainer` runs training with MLflow / W&B logging and early stopping.
6. Adapter weights are saved to `<output_dir>/<experiment>/adapter`.

## Merging for production inference

QLoRA adapters are trained against a 4-bit base, but for serving you typically
want the fused weights in bf16/fp16:

```bash
lora-finetune merge \
  --adapter outputs/qlora-llama3-8b-instruct/adapter \
  --output  outputs/qlora-llama3-8b-instruct/merged \
  --dtype   bfloat16
```

The merged directory is a drop-in for HF Transformers, vLLM or TGI.

## Post-training quantization

Compress the merged model further for faster inference:

```bash
# GPTQ (CUDA, INT4) — calibration-based, near-fp16 quality
python scripts/quantize_gptq.py --model outputs/merged --output outputs/merged-gptq-4bit --bits 4

# AWQ (CUDA, INT4) — activation-aware, slightly better than GPTQ at 4-bit
python scripts/quantize_awq.py --model outputs/merged --output outputs/merged-awq-4bit

# GGUF (llama.cpp, CPU/Metal) — for edge / Mac / Ollama
scripts/export_gguf.sh outputs/merged exports/llama3 q4_k_m
```

See [`docs/concepts.md`](docs/concepts.md) §7 for a full comparison.

## Multi-adapter serving

Host **one base model** and route requests to **N adapters** at runtime —
ideal for SaaS multi-tenant deployments. Adapter switching is free
(metadata-only) so per-tenant adapters live in the same pod.

```bash
lora-finetune serve \
  --model outputs/merged \
  --adapters support=outputs/support-adapter \
  --adapters sales=outputs/sales-adapter \
  --adapters legal=outputs/legal-adapter
```

Clients pick an adapter per request:
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Hi"}],
    "adapter": "support"
  }'
```

`GET /v1/adapters` lists what's mounted; `GET /health` reports adapter names.

## Observability

- **MLflow**: set `MLFLOW_TRACKING_URI`; `training.report_to: [mlflow]`.
- **Prometheus**: serving exposes `/metrics` with request counts, latency
  histograms and generated-token counters.
- **Structured logs**: JSON via `python-json-logger`, log level via `LOG_LEVEL`.

## Deployment

- **Docker**: `make docker-build` produces `lora-finetune:{train,serve}`.
- **docker-compose**: `docker compose --profile train up` runs MLflow + trainer.
- **Kubernetes**: `deploy/k8s/training-job.yaml` runs a one-shot Job on a GPU
  node; `deploy/k8s/inference-deployment.yaml` runs a scalable inference
  Deployment behind a Service with an HPA.
- **Airflow**: `deploy/airflow/training_dag.py` schedules weekly
  train → merge → rollout using `KubernetesPodOperator`.

## Testing

```bash
make test          # full suite (CPU)
ruff check src tests
mypy src
```

The CI workflow runs lint plus the CPU-safe subset of tests (config parsing,
templates, collator, metrics, API wiring, CLI) on every PR.

## Hardware notes

| Setup                | Base model        | VRAM peak | Notes                       |
| -------------------- | ----------------- | --------- | --------------------------- |
| QLoRA (4-bit NF4)    | Llama-3 8B        | ~12–16 GB | single 24 GB consumer GPU   |
| QLoRA (4-bit NF4)    | Llama-3 70B       | ~40–48 GB | A100 80 GB or 2× 24 GB      |
| LoRA (bf16)          | Mistral 7B        | ~24 GB    | A100 40 GB comfortable      |

Batch sizes assume `max_seq_length=2048` with gradient checkpointing.

## License

Apache-2.0.
