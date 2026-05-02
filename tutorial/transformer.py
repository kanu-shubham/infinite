"""
Tiny decoder-only transformer, built from scratch in ~200 lines.

A character-level GPT trained on a small text. Same architecture as nanoGPT,
intentionally minimal: classic LayerNorm, learned absolute positions, MHA,
GELU FFN. We will swap these for modern variants (RMSNorm, RoPE, SwiGLU, GQA,
KV cache) in follow-up commits.

Run:
    pip install torch
    python tutorial/transformer.py

Output: training loss curve + a sample of generated text.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
block_size    = 64       # context length
batch_size    = 32
n_embd        = 128      # d_model
n_head        = 4
n_layer       = 4
dropout       = 0.1
learning_rate = 3e-4
max_iters     = 3000
eval_interval = 500
device        = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# ---------------------------------------------------------------
# 1. Data + character tokenizer
# ---------------------------------------------------------------
text = (
    "First Citizen:\n"
    "Before we proceed any further, hear me speak.\n\n"
    "All:\n"
    "Speak, speak.\n\n"
    "First Citizen:\n"
    "You are all resolved rather to die than to famish?\n\n"
    "All:\n"
    "Resolved. resolved.\n\n"
    "First Citizen:\n"
    "First, you know Caius Marcius is chief enemy to the people.\n"
) * 400  # repeat so we have enough tokens to train on

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data, val_data = data[:n_split], data[n_split:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------
# 2. Multi-Head Self-Attention (causal)
# ---------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------
# 3. Feed-Forward Network
# ---------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# 4. Transformer block (pre-LayerNorm + residuals)
# ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------
# 5. Full GPT model
# ---------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_embd(idx)
        pos = self.pos_embd(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# ---------------------------------------------------------------
# 6. Training loop + sampling
# ---------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab={vocab_size}  params={n_params/1e6:.2f}M  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- sample ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=300)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
