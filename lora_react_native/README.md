# LoRA fine-tune: React Native code generation

End-to-end QLoRA fine-tune of an open coding LLM (default: `Qwen2.5-Coder-7B-Instruct`)
on a React Native instruction → code dataset.

## Why LoRA here

- The base model already knows JS/TS/React. You only need to bias it toward
  React Native idioms (`StyleSheet`, RN core components, navigation patterns,
  your component library). LoRA changes ~0.1–1% of params and is enough.
- QLoRA (4-bit base + LoRA on top) makes a 7B trainable on a single 24GB GPU.
- The adapter is ~50–200MB and can be hot-swapped per project / per design system.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Single source of truth for paths and hyperparameters. |
| `prepare_data.py` | Converts raw `{instruction, code}` JSONL into chat-formatted SFT data. Writes a synthetic seed if you don't have one yet. |
| `train.py` | QLoRA training loop using `trl.SFTTrainer` + `peft.LoraConfig`. |
| `infer.py` | Loads base + adapter and generates code from a prompt. |
| `merge_and_export.py` | Merges the adapter into the base weights for vLLM / TGI deployment. |

## Quickstart

```bash
pip install -r requirements.txt
huggingface-cli login          # if the base model is gated

python prepare_data.py         # builds data/train.jsonl + data/eval.jsonl
python train.py                # QLoRA fine-tune -> outputs/rn-coder-lora/
python infer.py "Build a RN profile screen with avatar, name, edit button."
```

## Bringing your own data

Replace `data/rn_seed.jsonl` with your real corpus. One JSON object per line:

```json
{"instruction": "Create a React Native ...", "code": "import React ..."}
```

Good sources:
- Your own private RN repos (extract component files + write instructions, or
  use a stronger model to back-translate code → instruction).
- Permissively licensed RN OSS projects (Expo examples, Ignite boilerplate).
- Snippets from your design system docs paired with usage prompts.

Aim for **a few thousand high-quality pairs** rather than tens of thousands of
noisy ones — for a narrow domain shift, quality dominates.

## Tuning knobs that matter

- `lora_r` / `lora_alpha`: bump to 32/64 if you're teaching a large proprietary
  component library, leave at 16/32 for general RN style.
- `lora_target_modules`: keep all linear layers (current default) — code
  quality drops noticeably if you only target attention projections.
- `max_seq_len`: raise to 4096 if your training examples are full screens.
- `lr`: 2e-4 is the QLoRA default; drop to 1e-4 if loss is unstable.

## Deployment

- Dev / experimentation: `infer.py` with the adapter loaded on top of the
  4-bit base.
- Production: run `merge_and_export.py` and serve the merged directory with
  vLLM (`vllm serve outputs/rn-coder-lora-merged`) or TGI for high throughput.
