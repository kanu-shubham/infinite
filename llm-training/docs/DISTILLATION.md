# Knowledge Distillation

Compress a large "teacher" model into a smaller "student" by training
the student to match the teacher's output distribution — not just the
ground-truth labels. Two modes are supported:

1. **Logit distillation** (this trainer): student matches the teacher's
   *full distribution* at every token, weighted by a temperature.
2. **Sequence-level distillation** (via `scripts/generate_teacher_data.py`):
   the teacher generates completions, and the student does plain SFT on
   them. Simpler; no teacher needed at train time; only requires the
   teacher's text outputs.

## When to use which

| Mode           | Pros                                               | Cons                                                           |
| -------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Logit          | Transfers the teacher's *uncertainty*; better quality | Teacher must fit in GPU memory during training                |
| Sequence-level | Simple; cacheable; works across tokenizers         | Loses information in the teacher's full distribution (dark knowledge) |

**Critical constraint for logit mode:** the student and teacher must
share the same tokenizer / vocabulary. Qwen → Qwen, Llama → Llama,
etc. Different tokenizers mean different logit dimensions — there's no
well-defined KL between them.

## Objective (logit mode)

Per-token loss is a convex combination of CE against hard labels and
KL against the temperature-softened teacher distribution:

$$
\mathcal{L} = \alpha_\text{ce} \cdot \mathcal{L}_\text{CE}(s, y)
            + \alpha_\text{kd} \cdot T^2 \cdot \text{KL}\!\left(
                \text{softmax}(s / T) \;\|\; \text{softmax}(t / T)
              \right)
$$

- $s, t$ — student and teacher logits
- $T$ — temperature (>1 softens the distributions, revealing dark knowledge)
- $T^2$ factor preserves gradient magnitude across temperatures
  (Hinton et al.)
- Only positions where the label is **not** `-100` contribute (i.e.,
  assistant response tokens — prompts are masked).

The `top_k_logits` option restricts the KL to the teacher's top-k
tokens per position. This can cut memory for large vocabularies and is
often *more* robust than the full-vocab KL because it ignores the
noisy long tail.

## Data

Same format as SFT (`messages` or `prompt`+`response`). The teacher
sees the identical tokenized input the student sees.

## Configuration

[`configs/distill/qwen_7b_to_0_5b.yaml`](../configs/distill/qwen_7b_to_0_5b.yaml). Key knobs:

| Field             | Notes                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| `student.*`       | Trained                                                               |
| `teacher.*`       | Frozen. Enable `load_in_4bit` if it won't fit otherwise               |
| `temperature`     | 1.0 = strict imitation; 2.0–4.0 = richer dark-knowledge transfer      |
| `alpha_ce`        | Weight on ground-truth CE — keeps the student grounded                |
| `alpha_kd`        | Weight on KL — drives the matching                                    |
| `top_k_logits`    | `50` trims tail noise; `null` uses full vocab (more memory)           |

Typical defaults: `T=2.0`, `alpha_ce=0.5`, `alpha_kd=0.5`.

## Running

```bash
# Logit distillation (teacher loaded in-process)
llm-train distill --config configs/distill/qwen_7b_to_0_5b.yaml
```

### Sequence-level distillation

Equivalent pipeline in two steps:

```bash
# 1. Generate teacher responses offline
python scripts/generate_teacher_data.py \
  --teacher Qwen/Qwen2.5-7B-Instruct \
  --prompts data/prompts.jsonl \
  --output  data/teacher.jsonl \
  --batch-size 16

# 2. SFT the student on the teacher's outputs
llm-train sft --config configs/sft/student_on_teacher.yaml
```

## Memory footprint

Logit distillation roughly doubles memory vs. SFT because the teacher's
activations are also in GPU memory during the forward. Mitigations:

- Quantize the teacher (`load_in_4bit: true`) — no gradients needed.
- Use `top_k_logits: 50` (instead of full vocab).
- Offload the teacher to CPU and run in chunks (not implemented here;
  requires a custom `accelerate` config).

## Evaluating

Compare student vs. teacher on held-out prompts:

```bash
llm-train evaluate --model outputs/distill_qwen_7b_to_0_5b \
                   --dataset data/examples/sft_sample.jsonl
```

Good distillation runs produce a student that:

- Matches the teacher's *style* (not just correctness)
- Approaches the teacher on in-domain tasks
- Runs 5–20× faster with 10–50× fewer parameters

## Common pitfalls

- **Tokenizer mismatch** — silent correctness failure. The dimensions
  happen to line up but the distributions are nonsense. The trainer
  does **not** check this; it's the caller's responsibility.
- **Teacher in bf16 but student in fp16** — precision mismatch can
  cause numerical issues. Keep both in bf16.
- **α_ce = 0** — pure imitation. Sometimes better, but if the teacher
  is imperfect, ground-truth CE acts as a regularizer.
- **Temperature too high** — flattens the teacher toward uniform;
  student learns nothing useful. Stay at T ≤ 4.
