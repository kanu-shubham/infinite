# Architecture

## Design principles

1. **Typed configs over CLI flags.** One YAML file per run. The file is
   the source of truth, versioned separately from the code, and
   snapshotted into every output directory.
2. **Hugging Face stack as the substrate.** `transformers`, `trl`,
   `peft`, `accelerate`, `datasets` — no custom abstractions over
   them. We add glue, not layers.
3. **One trainer per task.** SFT uses `transformers.Trainer`, DPO uses
   TRL's `DPOTrainer`, distillation subclasses `Trainer` with a custom
   `compute_loss`. Shared plumbing lives in `training/common.py`.
4. **Distributed execution via `accelerate`.** No hand-rolled DDP. Swap
   in DeepSpeed ZeRO-3, FSDP, or single-GPU by changing the
   `--config_file` passed to `accelerate launch`.
5. **Small surface, deep stack.** The Python package exposes only a
   handful of entry functions (`run_sft`, `run_dpo`, `run_distill`) —
   scripts and CLI are thin wrappers.

## Module map

```
llm_train/
├── config.py             # Discriminated-union Pydantic configs
├── data/
│   ├── loading.py        # Generic jsonl/parquet/hf loader
│   ├── sft.py            # Chat-template tokenization, completion-only masking
│   ├── dpo.py            # prompt/chosen/rejected normalization
│   └── distill.py        # Reuses SFT tokenization
├── models/loader.py      # AutoModel + tokenizer + (optional) bnb + LoRA
├── training/
│   ├── common.py         # Config → TrainingArguments, run snapshot
│   ├── collators.py      # Padding collator with label masking
│   ├── sft.py            # run_sft(cfg)
│   ├── dpo.py            # run_dpo(cfg)
│   └── distill.py        # LogitDistillationTrainer + run_distill(cfg)
├── evaluation/
│   ├── generate.py       # Chat-aware batched generation
│   └── harness.py        # JSONL-in/JSONL-out smoke eval
├── inference/
│   └── server.py         # FastAPI (OpenAI /v1/chat/completions subset)
├── utils/                # Logging (json/text), deterministic seeding
└── cli.py                # Typer CLI: sft / dpo / distill / evaluate / serve / merge-lora
```

## Config flow

```
configs/*.yaml
     │
     │  yaml.safe_load
     ▼
ConfigWrapper(cfg=...)                  # Pydantic discriminates on `task`
     │
     ▼
SFTConfig | DPOConfig | DistillConfig
     │
     ├─► snapshot_run() ──► <output_dir>/{resolved_config.yaml,
     │                                     source_config.yaml,
     │                                     run_meta.json}
     │
     ├─► load_model_and_tokenizer(cfg.model, cfg.tokenizer)
     │
     ├─► apply_lora(model, cfg.lora)
     │
     ├─► build_<task>_datasets(...)
     │
     └─► build_hf_training_args(cfg.training, cfg.optimizer) ─► HF Trainer / DPOTrainer / DistillationTrainer
```

## Distillation data flow

```
                 ┌───────────────────┐
     batch ────► │ student.forward() │ ─► student_logits
                 └───────────────────┘        │
                                              │  CE(shift_logits, shift_labels)
                                              │  → loss_ce
                                              ▼
                 ┌───────────────────┐
     batch ────► │ teacher.forward() │ ─► teacher_logits     (inference_mode, frozen)
                 └───────────────────┘        │
                                              │  soften by T, optional top-k
                                              ▼
                                 KL(student_T || teacher_T) · T²
                                              │
                                              │  label_mask (ignore prompt + padding)
                                              ▼
                        loss = α_ce · loss_ce + α_kd · loss_kd
```

## Failure modes & invariants

| Invariant                                                              | Where enforced                       |
| ---------------------------------------------------------------------- | ------------------------------------ |
| No more than one of `load_in_4bit` / `load_in_8bit`                    | `ModelConfig._exclusive_quant`       |
| Config has a `task:` key                                               | `load_config()`                      |
| Distillation: `alpha_ce + alpha_kd > 0`                                | `DistillConfig._weights_sum`         |
| Final SFT message is `role=assistant` when `completion_only_loss=True` | `data/sft.py::_encode_example`       |
| DPO row has `prompt`, `chosen`, `rejected`                             | `data/dpo.py::_normalize_row`        |
| Tokenizer has a pad token                                              | `models/loader.py::load_tokenizer`   |

## Extending

- **New loss variant** → add a field to `DPOConfig.loss_type` (TRL already
  accepts it) or subclass `LogitDistillationTrainer`.
- **New data source** → add a branch in `data/loading.py`.
- **Custom callback** (e.g. generation during eval) → pass
  `callbacks=[...]` to the trainer constructor in `training/sft.py`.
- **New model family with non-standard chat template** → set
  `tokenizer.chat_template` in the config.
