# Transformer Architecture: Deep Dive

A practitioner-oriented walk through the core machinery of modern transformer
LLMs, with an emphasis on the trade-offs that surface in production.

---

## 1. Attention

### 1.1 Scaled Dot-Product Attention (recap)

Given input projections `Q ∈ R^{n×d_k}`, `K ∈ R^{n×d_k}`, `V ∈ R^{n×d_v}`:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

The `√d_k` scaling prevents the pre-softmax logits from growing with dimension
and pushing softmax into low-gradient saturation. Causal (decoder) attention
adds a mask that sets future positions to `-∞` before softmax, so position `t`
only attends to `≤ t`.

Compute is `O(n² · d)` and memory is `O(n²)` for the attention matrix — the
quadratic term that dominates long-context cost.

### 1.2 Multi-Head Attention (MHA)

Run `h` parallel attention heads, each with its own `(W_Q, W_K, W_V)` of width
`d_model / h`, then concatenate and project with `W_O`. Heads specialize:
empirical probing finds positional, syntactic, and rare-token heads. Cost:

- Params per layer: `4 · d_model²` (Q, K, V, O projections).
- KV cache per token per layer: `2 · h · d_head = 2 · d_model` values.

For a 70B model with `d_model=8192`, `n_layers=80`, FP16, that's `~2.5 MB of
KV per token` — the dominant memory cost at long context.

### 1.3 Multi-Query Attention (MQA)

Share **one** K and one V across all heads; keep per-head Q. KV cache shrinks
by a factor of `h` (often 32–128×). Quality degrades modestly but noticeably,
especially on tasks requiring fine-grained retrieval. Used in PaLM and early
Falcon.

### 1.4 Grouped-Query Attention (GQA)

Compromise: split heads into `g` groups, share K/V within a group. With `g=8`
on 64 heads you get 8× KV reduction at near-MHA quality. This is the dominant
choice today (Llama 2/3, Mistral, Qwen 2, Gemma).

| Variant | KV size  | Quality | Typical use            |
| ------- | -------- | ------- | ---------------------- |
| MHA     | `h · d`  | Best    | Small models, research |
| GQA     | `g · d`  | ~MHA    | Modern open LLMs       |
| MQA     | `1 · d`  | Lower   | Throughput-critical    |

**Production implication:** GQA changes weight layout — you can't naively
upcycle MHA checkpoints. There is a known "uptraining" recipe (mean-pool the
shared heads, fine-tune ~5% of original tokens) that recovers most of the gap.

### 1.5 Flash / fused attention

These are not architectural variants but **kernel** changes: tile Q/K/V into
SRAM, fuse softmax with matmul, recompute on backward to avoid materializing
the `n × n` matrix. FlashAttention-2/3 give ~2–4× speedup and shrink activation
memory from `O(n²)` to `O(n)`. They are mathematically identical to vanilla
attention; you adopt them for free at training and inference time.

---

## 2. KV Cache

At decode time, for each new token you only need to compute its Q, then attend
over **all previously cached K/V**. Without a cache you redo `O(n)` work per
step and total decode is `O(n²)`; with a cache it's `O(n)` work per step,
`O(n²)` total but with a much smaller constant — and the bottleneck shifts
from FLOPs to **memory bandwidth**.

### 2.1 Sizing

```
kv_bytes = 2 (K and V)
         · n_layers
         · n_kv_heads          # = h for MHA, g for GQA, 1 for MQA
         · d_head
         · seq_len
         · dtype_bytes         # 2 for FP16/BF16, 1 for FP8/INT8
```

For Llama-3-70B (GQA, `n_kv_heads=8`, `d_head=128`, `n_layers=80`) at 8K
context in BF16: `2·80·8·128·8192·2 ≈ 2.7 GB per sequence`. At 128K context:
`~42 GB per sequence` — larger than the model weights' GPU footprint per shard.

### 2.2 Production techniques

- **Paged KV cache (vLLM)**: allocate cache in fixed-size blocks (à la OS
  paging) instead of contiguous per-request buffers. Eliminates the
  fragmentation that wastes 60–80% of cache memory under variable-length
  serving. Enables copy-on-write for shared prefixes.
- **Prefix caching / radix attention**: hash and reuse KV for shared prompts
  (system prompt, few-shot examples, RAG context). Free latency win for any
  workload with prompt overlap.
- **KV quantization**: INT8 and FP8 KV caches are now standard; INT4 KV is
  viable with per-head scales but degrades long-context recall.
- **Cache offload / eviction**: H2O, StreamingLLM, and SnapKV evict
  low-importance tokens; useful past 100K context where KV no longer fits.
- **Speculative decoding**: a small draft model proposes `k` tokens; the big
  model verifies in one forward pass. KV cache management for the draft is
  the subtle part — you must roll back on rejection.

### 2.3 Continuous batching

The KV cache is what makes naive batching painful: requests have different
lengths and finish at different times. Continuous batching (Orca, vLLM)
schedules at the token level — finished sequences are evicted and new ones
slot into their KV blocks immediately. Throughput gains of 10–20× over static
batching are typical.

---

## 3. Positional Encodings

Self-attention is permutation-equivariant; position must be injected
explicitly.

### 3.1 Sinusoidal / learned absolute (original)

Add a fixed (or learned) vector to the token embedding. Simple but doesn't
extrapolate beyond training length and entangles position with content.

### 3.2 RoPE (Rotary Position Embedding)

Rotate each pair of dimensions in Q and K by an angle proportional to position
`m`:

```
q_m = R_m · q,   k_n = R_n · k
qᵀ k after rotation depends only on (m − n), giving a relative encoding.
```

Frequencies are geometric: `θ_i = base^(-2i/d)`, base typically `10000`.
RoPE is now the default (Llama, Mistral, Qwen, DeepSeek, Gemma).

**Context extension.** RoPE doesn't generalize past training length zero-shot,
but you can extend it cheaply:

- **Position Interpolation (PI)**: divide positions by a factor `s`. Crude
  but works with light fine-tuning.
- **NTK-aware scaling**: scale the RoPE base instead of positions, preserving
  high-frequency information that PI smears.
- **YaRN**: per-frequency interpolation strategy (high freqs preserved, low
  freqs interpolated, with a temperature correction). State of the art for
  4–8× extension.
- **LongRoPE**: search per-dimension scaling factors; pushes to 2M+ tokens.

### 3.3 ALiBi (Attention with Linear Biases)

Add a position-dependent linear penalty to attention logits:
`logit_{ij} += -m_h · |i − j|`, with a per-head slope `m_h`. No embeddings;
extrapolates further than learned schemes out of the box. Used in MPT and
BLOOM. Quality slightly trails RoPE on most evals; rarely chosen for new
models.

### 3.4 What about NoPE?

Decoder-only causal models can learn position implicitly from the mask.
"NoPE" works surprisingly well in small ablations but loses to RoPE at scale
and on long context. Useful as a baseline, not a deployment choice.

---

## 4. Layer Norm Placement

The layer-norm (or RMSNorm) location around the attention / FFN sublayers
matters more than it sounds.

### 4.1 Post-LN (original Vaswani)

```
x ← LN( x + Sublayer(x) )
```

Norm is **outside** the residual. Beautifully balanced at convergence but
empirically unstable — you need a learning-rate warmup and even then deep
post-LN networks (>~12 layers) commonly diverge in fp16.

### 4.2 Pre-LN (modern default)

```
x ← x + Sublayer( LN(x) )
```

Norm is **inside** the residual; the residual stream is never normalized
directly. Trains stably without warmup tricks at 100+ layers. Drawback: the
residual stream can grow without bound across depth, which interacts badly
with quantization. All major modern LLMs are pre-LN (or a variant).

### 4.3 RMSNorm

Drop the mean-centering and bias from LayerNorm:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · g
```

~7–15% faster than LayerNorm with no quality loss; standard in Llama,
Mistral, Gemma. The lack of bias/recentering also plays nicely with FP8.

### 4.4 Sandwich / DeepNorm / Peri-LN

Sandwich-LN (Cogview), DeepNorm (GLM-130B), and Peri-LN (Apple) add an extra
norm — typically on the sublayer output before the residual add — to recover
post-LN's signal balance without its instability. These matter for very deep
or very wide training runs and for FP8/INT8 stability; they're invisible at
inference.

### 4.5 QK-Norm

Normalize Q and K **before** computing attention logits. Tames the occasional
massive activation that destabilizes long training runs (seen in GLM, Gemma 2).
Cheap and increasingly common.

---

## 5. Decoder-only vs Encoder-Decoder

### 5.1 The three families

- **Encoder-only** (BERT, RoBERTa): bidirectional attention, MLM objective.
  Strong for embeddings and classification; can't generate naturally.
- **Encoder-decoder** (T5, BART, original Transformer): bidirectional encoder
  + causal decoder with cross-attention. Strong for seq2seq (translation,
  summarization).
- **Decoder-only** (GPT, Llama, Claude, Gemini): causal attention only,
  next-token-prediction objective.

### 5.2 Why decoder-only won for general LLMs

| Factor                | Decoder-only                  | Encoder-decoder                       |
| --------------------- | ----------------------------- | ------------------------------------- |
| Training objective    | Single (next-token)           | Two halves (encoder MLM + decoder LM) |
| Compute efficiency    | Every token contributes loss  | Encoder tokens contribute via decoder |
| Few-shot / in-context | Native — prompt is just text  | Awkward; must fit encoder window      |
| KV cache              | One cache                     | Two caches; cross-attention adds K,V  |
| Scaling               | Cleanest scaling laws         | More architectural knobs              |
| Streaming             | Token-by-token natural        | Encoder must finish first             |

For a fixed compute budget, decoder-only matches or beats enc-dec on
generation tasks once instruction tuning is in the mix (Wang et al. 2022,
"What Language Model Architecture and Pretraining Objective Work Best for
Zero-Shot Generalization?").

### 5.3 Where encoder-decoder still wins

- **Tasks with a clear, bounded input → output mapping** where bidirectional
  encoding helps (translation, dense summarization, code repair). T5 and
  Flan-T5 are still competitive per-FLOP here.
- **Latency-critical generation with long inputs** — the encoder runs once;
  the decoder is small. NLLB, Whisper (audio enc, text dec), and most ASR/MT
  systems use this.
- **Diffusion-style / parallel decoding**: the encoder context is fixed.

### 5.4 Prefix-LM as a middle ground

Apply bidirectional attention to the prefix, causal to the continuation, in a
single decoder. Captures most of the encoder benefit without two stacks. Used
in UL2 / PaLM-2's training mix.

---

## 6. Tokenization

Tokenization sits **outside** the model but quietly determines a large
fraction of cost, latency, multilingual quality, and several whole categories
of bugs.

### 6.1 BPE (Byte-Pair Encoding)

Start from a base alphabet (bytes, in modern "byte-level BPE"), greedily merge
the most frequent adjacent pair, repeat until you hit the target vocab size.
GPT-2/3/4, Llama, Mistral, Claude all use byte-level BPE variants.

Properties:

- **No OOV** — every byte sequence is encodable.
- **Deterministic encoding** given the merge table.
- **Greedy and left-to-right** — locally suboptimal segmentations are common.

### 6.2 WordPiece

Like BPE but the merge criterion maximizes training-corpus likelihood instead
of pair frequency. Used by BERT. Practically very similar to BPE for users.

### 6.3 SentencePiece (Unigram LM)

Start from a large candidate vocab, iteratively prune tokens that least hurt
corpus likelihood under a unigram model. Encoding is **probabilistic** —
multiple segmentations exist and you can sample them ("subword regularization"),
which acts as data augmentation. Used by T5, XLNet, ALBERT, Gemma, mT5.

SentencePiece (the library) also handles whitespace as a regular character
(`▁`), which is what makes tokenizer round-tripping reliable across
languages without explicit pre-tokenization.

### 6.4 Production implications

**Cost & latency.** Tokens are the unit of pricing, KV cache, and FLOPs.
A poorly chosen tokenizer for your domain inflates token counts by 1.5–4×.

- English: ~0.75 tokens/word for Llama/GPT-4 tokenizers.
- Code: indentation and identifiers cost more; tokenizer choice swings cost
  by 30–50% on Python/JS-heavy traffic.
- CJK / Indic / Arabic: legacy GPT-2 BPE used 2–4× more tokens per character
  than English. Modern multilingual tokenizers (Gemma's 256k, GPT-4o's 200k,
  Llama-3's 128k) close most of the gap.

**Vocabulary size trade-offs.** Bigger vocab → fewer tokens per text →
cheaper inference, but the embedding and unembedding matrices grow linearly
in vocab size. At 128k vocab and `d_model=8192`, the embedding alone is ~2 GB
in BF16 and dominates small-model parameter counts.

**Subtle bugs.**

- **Leading-space tokens.** `"hello"` and `" hello"` are different tokens.
  Stripping or adding whitespace mid-prompt silently changes semantics.
- **Glitch / unreachable tokens.** Some tokens (`SolidGoldMagikarp` famously)
  exist in the vocab but were never trained — they produce undefined behavior.
  Audit your vocab against training-data frequency before deploying.
- **Mid-token sampling at the boundary** during streaming / speculative
  decoding requires careful detokenization; partial UTF-8 sequences are a
  classic source of mojibake.
- **Prompt-injection via tokenization.** Homoglyphs and zero-width characters
  often tokenize to surprising sequences and bypass naive content filters.
- **Numbers.** Inconsistent digit tokenization is a major reason small models
  fail at arithmetic. Llama-3 forces per-digit tokenization for digits, which
  measurably helps math.
- **Tokenizer drift.** If you fine-tune with a tokenizer that doesn't exactly
  match the base model's, you get silent off-by-one quality regressions. Pin
  the tokenizer hash alongside the model weights.

**Domain-specific vocab.** For code, biomedical text, or non-Latin scripts,
training a small **tokenizer extension** (a few thousand merged tokens added
to the base vocab, with new embedding rows initialized from mean of subwords)
recovers most of the per-token-count gap without touching the rest of the
model.

---

## 7. Putting it together — a modern decoder block

The pieces above compose into the canonical 2024–2026 LLM block:

```
x ← x + Attention( RMSNorm(x), RoPE, GQA )
x ← x + FFN(      RMSNorm(x), SwiGLU )
```

with optional QK-norm inside attention, FlashAttention kernels at compute
time, paged GQA KV cache at serve time, and a 128k–256k byte-level BPE (or
unigram) tokenizer in front. Each choice is independently swappable, which is
why the same recipe scales from 1B to 400B+ with surprisingly little
re-tuning.

---

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Press et al., *Train Short, Test Long: Attention with Linear Biases* (2021)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- Dao, *FlashAttention-2* (2023); Shah et al., *FlashAttention-3* (2024)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023)
- Peng et al., *YaRN: Efficient Context Window Extension* (2023)
- Wang et al., *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?* (2022)
- Kudo, *Subword Regularization* (2018) — SentencePiece unigram
