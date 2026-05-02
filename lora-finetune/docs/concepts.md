# Concepts: Fine-tuning, PEFT, and Quantization

A practitioner's guide to the techniques implemented in this project.
Read this before tuning configs — most "training mysteriously fails" issues
trace to violating one of the principles below.

---

## 1. Why fine-tune at all?

A pre-trained LLM has learned the general statistics of language and a broad
slice of world knowledge. Fine-tuning specializes it to:

- **Format / style** (your support tone, JSON-only outputs, brand voice)
- **Domain** (medical, legal, internal company jargon)
- **Task** (summarize tickets, route emails, classify intents)

Three approaches exist:

| Approach              | Trains              | Pros                         | Cons                          |
| --------------------- | ------------------- | ---------------------------- | ----------------------------- |
| **Prompting / RAG**   | nothing             | Zero training cost           | Long prompts, no behavior change |
| **PEFT**              | <1% of params       | Cheap, modular, no forgetting | Slightly weaker than full FT |
| **Full fine-tuning**  | 100% of params      | Maximum quality              | Expensive, forgetting risk    |

This project optimizes for the middle row.

---

## 2. The math of LoRA

### 2.1 The pre-trained weight

Every linear layer in a Transformer applies `y = W·x` where `W ∈ ℝ^(d_out × d_in)`.
For a 7B model with `d_in = d_out = 4096`, each `W` has ~17M parameters, and
there are dozens of them. Updating all of them is what makes full FT
expensive.

### 2.2 The LoRA hypothesis

**Hu et al. (2021)** observed that the *update* `ΔW = W_finetuned − W_pretrained`
during fine-tuning is empirically **low-rank**. Most useful task-adaptation
lives in a tiny subspace.

So, instead of learning `ΔW` directly, parameterize it as:

```
ΔW = B · A           where  A ∈ ℝ^(r × d_in)
                            B ∈ ℝ^(d_out × r)
                            r ≪ min(d_in, d_out)
```

The forward pass becomes:

```
y = W·x + (α/r) · B·A·x
    └────┘   └─────────┘
    frozen    trainable
```

`(α/r)` is a fixed scaling factor that decouples the **learning rate** from
the **rank** — you can change `r` without re-tuning LR.

### 2.3 Initialization

- `A` ~ `kaiming_uniform`
- `B` = `0`

So at step 0, `B·A·x = 0` — the model is identical to the base. Training
gently steers it away.

### 2.4 Parameter count

For one linear layer:

- Full FT: `d_in · d_out` params
- LoRA:    `r · (d_in + d_out)` params

For `d = 4096, r = 16`: full = 16.8M, LoRA = 131K → **128× fewer**.

### 2.5 Inference

After training, you can either:

1. Keep the adapter separate: `y = W·x + (α/r)·B·A·x`. Two matmuls per layer,
   slight overhead, but you can hot-swap adapters.
2. **Merge**: compute `W' = W + (α/r)·B·A` once and store. Single matmul, zero
   runtime overhead. (`scripts: lora-finetune merge ...`)

---

## 3. Choosing rank `r` and `alpha`

### Rule of thumb
```
alpha = 2 · r       # the effective learning rate for the adapter
```

This is the most common default; it makes the LoRA update magnitude
comparable to vanilla SGD on the full weight.

### Picking `r`

| `r`     | Use when                                                       |
| ------- | -------------------------------------------------------------- |
| **4–8** | Tone/format adaptation, small dataset (< 5k examples)          |
| **16**  | Default for instruction-following, balanced datasets           |
| **32–64** | Domain shift (medical, legal), >50k examples                 |
| **128+** | Large behavioral change, lots of data; consider full FT instead |

Higher `r` → more capacity but more risk of overfitting and forgetting.

### RSLoRA scaling

With **rank-stabilized LoRA**, scaling becomes `α / √r` instead of `α / r`.
This lets you push `r` higher without instability. Set `use_rslora: true`.

### DoRA — direction + magnitude

DoRA decomposes the weight into a unit-norm direction and a magnitude scalar,
training both. Often improves quality at fixed `r`, especially for low ranks
(`r = 4–16`). Cost: ~25% more memory than LoRA, ~10% slower training. Set
`use_dora: true`.

---

## 4. Target modules — which layers to LoRA-fy?

A Transformer block has these linear layers:

- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP**: `gate_proj`, `up_proj`, `down_proj` (Llama-style)
  or `fc1`, `fc2` (GPT-style)
- **Embeddings**: typically frozen

### Common policies

| Strategy                      | Modules targeted                                           | Notes                       |
| ----------------------------- | ---------------------------------------------------------- | --------------------------- |
| **`q_proj`, `v_proj` only**   | the original LoRA paper's setup                            | Cheapest, weakest           |
| **`q,k,v,o`**                 | full attention                                             | Good middle ground          |
| **`q,k,v,o,gate,up,down`**    | attention + MLP                                            | **Best quality, ~3× params**     |
| **`all-linear`**              | every nn.Linear in the model                               | Simplest spec, equivalent to above for most arch    |

**Rule:** when in doubt, target everything (`all-linear`). The overhead is
trivial compared to the base model, and quality usually wins.

### Caveats

- Don't include layer-norm or embedding layers in `target_modules` — those
  aren't `nn.Linear` and PEFT will skip them, or worse, error.
- Small models (< 1B) sometimes regress with LoRA on MLP — try attention-only first.

---

## 5. Other PEFT methods

### 5.1 Prefix Tuning

**Idea:** prepend `n` trainable continuous vectors (the "prefix") to the
key and value of every attention layer. The base model is fully frozen; only
the prefix vectors and (optionally) a projection MLP are trained.

```
K' = concat(P_k, K)     # P_k ∈ ℝ^(num_virtual_tokens × d_k)
V' = concat(P_v, V)
attention(Q, K', V')
```

**Trainable params:** `2 · n_layers · n · d_model` (tiny — typically <0.1%).

**When to use:** lots of similar tasks, want one adapter per task with
minimum disk footprint. Often weaker than LoRA for hard tasks.

### 5.2 P-Tuning v2

Like prefix tuning, but a small **prompt encoder** (MLP or LSTM) generates
the virtual tokens from a learned embedding table. The reparameterization
helps optimization for some tasks.

**Trainable params:** prefix params + the encoder (still tiny).

**When to use:** classification or short-answer tasks. Less popular than LoRA
for chat/instruction tuning.

### 5.3 (IA)³

**Infused Adapter by Inhibiting and Amplifying Inner Activations.**
Learns three rescaling vectors per layer:

```
K' = K ⊙ ℓ_k        # element-wise scale, ℓ_k ∈ ℝ^d
V' = V ⊙ ℓ_v
FFN(x) = (W_down · GELU(W_up · x)) ⊙ ℓ_ff
```

That's it — three vectors. **Trainable params:** ~0.01% of base. Initialize
to 1 (so the model starts identical to base).

**When to use:** simple style/tone shifts; constrained budgets. Surprisingly
strong for low-data regimes; weak for big behavioral changes.

### 5.4 Comparison

| Method      | Trainable %    | Quality      | Hot-swap | Notes                                |
| ----------- | -------------- | ------------ | -------- | ------------------------------------ |
| LoRA        | 0.1–1%         | ★★★★         | Yes      | Default choice                       |
| QLoRA       | 0.1–1%         | ★★★★         | Yes      | LoRA + 4-bit base; 70B on 1 GPU      |
| DoRA        | 0.1–1.2%       | ★★★★+        | Yes      | LoRA-but-better at low rank          |
| Prefix      | 0.05–0.1%      | ★★★          | Yes      | Modular, weakest at hard tasks       |
| P-Tuning v2 | 0.05–0.1%      | ★★★          | Yes      | Better than prefix on some tasks     |
| (IA)³       | 0.01%          | ★★★          | Yes      | Tiniest; great for low data          |
| Full FT     | 100%           | ★★★★★        | No       | Most expensive, strongest            |

---

## 6. Catastrophic forgetting

When you fine-tune, the model can lose pre-training capabilities — the
classic "I trained on customer-support chats and now my model can't add
two numbers."

### How PEFT helps

Frozen base weights → general capabilities are preserved by construction.
LoRA can over-write the response *style* but the underlying knowledge stays
addressable. This is the single biggest reason to prefer PEFT over full FT.

### When forgetting still happens with PEFT

- **High `r` + small dataset**: too much capacity overfits the narrow domain
- **High learning rate**: pushes the adapter into a region that suppresses
  base behaviors
- **Long training** without held-out eval: classic overfitting

### Mitigations

1. **Mix in general data.** Add 10–20% of a high-quality general
   instruction dataset (UltraChat, OpenOrca) to your task data.
2. **Eval on held-out general prompts** *and* task prompts — track both.
3. **Lower `r`, shorter training, smaller LR.**
4. **Use `modules_to_save`** sparingly (it un-freezes more weights, increases
   forgetting risk).

### When forgetting is acceptable

A task-specific full-FT model that does *only* one thing (e.g., classify
support tickets) is fine — it's not meant to be a general assistant.

---

## 7. Quantization landscape

Quantization compresses weights from fp16/bf16 (2 bytes/param) down to 8, 4,
or even 2 bits. There are three different *moments* you might quantize:

| When                   | What                                              | Used for             |
| ---------------------- | ------------------------------------------------- | -------------------- |
| **At training** (LoRA) | bitsandbytes 4-bit base + LoRA fp16 adapter (QLoRA) | Cheap fine-tuning    |
| **Post-training**      | GPTQ, AWQ, GGUF                                   | Fast inference       |
| **Mixed precision**    | bf16/fp16 forward + fp32 master weights           | All training         |

### 7.1 bitsandbytes (training-time)

**4-bit NF4 (NormalFloat 4):** information-theoretically optimal codebook for
*normally-distributed* weights. Plus "double quantization" of the
quantization constants themselves saves another ~0.4 bits/param.

Used in QLoRA: the base model lives at 4-bit, gradients flow through a tiny
fp16 dequantization shim, and the LoRA adapter is fp16. Memory drops ~4×;
quality drop is typically <1%.

**Set:**
```yaml
quantization:
  enabled: true
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
```

bitsandbytes also has 8-bit (`load_in_8bit: true`) — less aggressive, ~2×
memory savings, useful when 4-bit hurts your task.

### 7.2 GPTQ (post-training)

**Idea:** for each layer, find INT4/INT3 weights that minimize the
reconstruction error of `W·x` on a small calibration set, processed
column-by-column with a Hessian-based update.

- Quality very close to fp16
- Fast inference (no per-layer dequant overhead like bnb)
- Requires calibration data (~128 samples is enough)
- One-time conversion: ~10–30 min for 7B

```bash
python scripts/quantize_gptq.py --model outputs/merged --output outputs/merged-gptq-4bit --bits 4
```

### 7.3 AWQ (post-training)

**Activation-aware Weight Quantization.** Looks at which weight channels
have the largest activation magnitudes and *protects* those (~1% of channels)
from being quantized. Slightly better quality than GPTQ at 4-bit, similar
speed.

```bash
python scripts/quantize_awq.py --model outputs/merged --output outputs/merged-awq-4bit
```

### 7.4 GGUF (post-training, llama.cpp)

GGUF is the file *format* used by llama.cpp for CPU/Metal/Vulkan inference.
It supports a family of quantizations: `q4_0`, `q4_k_m` (most popular),
`q5_k_m`, `q8_0`, `f16`. The "K" quantizations group small blocks and use
mixed precision per-block.

Use it when you want to deploy to:
- CPUs (laptops, edge devices)
- Apple Silicon via Metal
- Ollama, LM Studio, llama-server

```bash
scripts/export_gguf.sh outputs/merged exports/llama3-q4 q4_k_m
```

### 7.5 Quick decision matrix

| Goal                                    | Use                       |
| --------------------------------------- | ------------------------- |
| Fine-tune big model on small GPU        | **QLoRA (bnb 4-bit NF4)** |
| Serve on CUDA, fastest inference        | **GPTQ or AWQ INT4**      |
| Serve on CPU / Mac / edge               | **GGUF (q4_k_m)**         |
| Match training accuracy exactly         | **bf16 (no quant)**       |
| Fit borderline model in memory          | **bnb 8-bit**             |

### 7.6 Mixed precision training (bf16/fp16)

Orthogonal to weight quantization: keep an fp32 master copy of the weights
and optimizer state, but compute forward/backward in bf16/fp16 to halve
memory bandwidth.

**bf16 vs fp16:**
- **bf16** has the same exponent range as fp32 → no loss scaling, no NaN
  issues. **Strongly preferred** if your GPU supports it (Ampere+, A100, H100,
  RTX 30/40 series).
- **fp16** has more mantissa bits but smaller range → needs dynamic loss
  scaling, occasional overflow. Use only on older GPUs (V100, T4).

We default to `bf16: true`.

---

## 8. Decision flowchart

```
                      ┌─────────────────────┐
                      │  Need to fine-tune  │
                      │       an LLM        │
                      └──────────┬──────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
   Tiny data + tiny       Adapt one model         Big behavior change,
   style change           per use case            strong domain shift,
         │               (most cases)             unique single-purpose model
         │                       │                       │
         ▼                       ▼                       ▼
     [ (IA)³ ]              [ LoRA / QLoRA ]        [ Full FT ]
                                  │
                                  ▼
                          Does base fit in
                          one GPU at fp16?
                                  │
                          ┌───────┴───────┐
                         Yes              No
                          │                │
                          ▼                ▼
                       [ LoRA ]         [ QLoRA ]


                  ┌─────────────────────────────┐
                  │  Trained, ready to deploy?  │
                  └──────────────┬──────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
        Need maximum         CUDA inference,    Edge / CPU /
        flexibility          high QPS           Apple Silicon
        (multi-tenant)            │                   │
              │                   │                   │
              ▼                   ▼                   ▼
     [ Multi-adapter      [ Merge → GPTQ      [ Merge → GGUF
       serving ]            or AWQ INT4 ]       q4_k_m ]
```

---

## 9. Cost / quality cheat sheet

For a 7B-class base model, single A100-40GB, ~10k training examples:

| Setup           | VRAM peak | Train time | Adapter size | Quality |
| --------------- | --------- | ---------- | ------------ | ------- |
| Full FT (bf16)  | ~140 GB*  | n/a       | 14 GB        | ★★★★★   |
| Full FT + ZeRO-3 | ~40 GB    | 6 h       | 14 GB        | ★★★★★   |
| LoRA (bf16)     | ~24 GB    | 2 h        | 50 MB        | ★★★★    |
| QLoRA (4-bit)   | ~12 GB    | 3 h        | 50 MB        | ★★★★    |
| Prefix          | ~22 GB    | 1.5 h      | 5 MB         | ★★★     |
| (IA)³           | ~22 GB    | 1.5 h      | 0.5 MB       | ★★★     |

\* Without ZeRO; needs multi-GPU or sharding to fit.

---

## 10. Further reading

- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of LLMs* (2021)
- **QLoRA** — Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- **DoRA** — Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024)
- **GPTQ** — Frantar et al., *GPTQ: Accurate Post-Training Quantization* (2022)
- **AWQ** — Lin et al., *AWQ: Activation-aware Weight Quantization* (2023)
- **(IA)³** — Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (2022)
- **Prefix tuning** — Li & Liang, *Prefix-Tuning* (2021)
- **P-Tuning v2** — Liu et al., *P-Tuning v2* (2021)
