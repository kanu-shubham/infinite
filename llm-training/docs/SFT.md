# Supervised Fine-Tuning (SFT)

SFT teaches a pretrained base model to follow a particular **style** or
**task format**. It's the first post-training step before any preference
alignment.

## Objective

Standard next-token cross-entropy, but with the prompt tokens masked so
loss is only backpropagated over the **assistant response** tokens:

$$
\mathcal{L}_{\text{SFT}} = - \sum_{t \in \text{response}} \log p_\theta(y_t \mid y_{<t})
$$

This matches `completion_only_loss: true` in the config. It's the default
because training on prompt tokens biases the model toward reproducing
prompts rather than responding to them.

## Data format

Two accepted row shapes in JSONL / Parquet / HF datasets:

**Chat format (preferred):**
```json
{"messages": [
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "user", "content": "What is RAG?"},
  {"role": "assistant", "content": "Retrieval-augmented generation..."}
]}
```

**Plain prompt/response:**
```json
{"prompt": "What is RAG?", "response": "Retrieval-augmented generation..."}
```

The final message must be `assistant` when `completion_only_loss: true`.

## Configuration

See [`configs/sft/qwen_0_5b.yaml`](../configs/sft/qwen_0_5b.yaml). Key knobs:

| Field                            | Effect                                                         |
| -------------------------------- | -------------------------------------------------------------- |
| `lora.enabled`                   | LoRA if true (recommended); full FT otherwise                  |
| `lora.r` / `lora.alpha`          | LoRA rank / scaling — `r=16, alpha=32` is a solid default      |
| `completion_only_loss`           | Mask prompt tokens in the loss                                 |
| `tokenizer.model_max_length`     | Truncation window                                              |
| `optimizer.learning_rate`        | `2e-4` for LoRA, `1e-5` – `5e-5` for full FT                   |
| `training.gradient_checkpointing`| Saves activations memory; ~20–30% slower                       |
| `model.load_in_4bit`             | QLoRA — enables 7B+ training on a 24 GB GPU                    |

## Running

```bash
# Single GPU
llm-train sft --config configs/sft/qwen_0_5b.yaml

# 8-GPU DeepSpeed ZeRO-3
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_sft.py --config configs/sft/qwen_0_5b.yaml
```

## Outputs

After training `<output_dir>` contains:

- `resolved_config.yaml` — the exact config that was executed
- `source_config.yaml` — a copy of the input YAML
- `run_meta.json` — start timestamp, task type
- Standard HF checkpoint files (`config.json`, `*.safetensors`, tokenizer)
- `trainer_state.json` + `all_results.json`

To merge a LoRA adapter for inference:

```bash
llm-train merge-lora \
  --adapter outputs/sft_qwen_0_5b \
  --base   Qwen/Qwen2.5-0.5B-Instruct \
  --output outputs/sft_qwen_0_5b_merged
```

## Common pitfalls

- **Loss near zero immediately** — you probably didn't mask the prompt;
  the model is memorizing prompts.
- **Loss diverges** — LR too high (typical for full FT), or `bf16=false`
  on fp16-unfriendly hardware.
- **OOM on long sequences** — enable `gradient_checkpointing`, lower
  `per_device_train_batch_size`, raise `gradient_accumulation_steps`.
- **Chat template mismatch** — when porting data across models, make
  sure the tokenizer's `chat_template` renders the expected string.
