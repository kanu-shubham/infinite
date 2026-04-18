"""
Project 17: Transformers & Attention from Scratch
===================================================
Understand the mechanism behind GPT, BERT, and all modern AI —
by building it from scratch in PyTorch.

What you'll learn:
- Self-attention: how tokens "talk" to each other
- Multi-head attention: attending to different aspects simultaneously
- Positional encoding: injecting order into attention
- Full Transformer encoder block
- Using HuggingFace Transformers for sequence classification
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Part 1: Self-Attention from Scratch ───────────────────────────────────

class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention.

    For each token, compute how much it should "attend" to every other token.

    Q (Query):  "What am I looking for?"
    K (Key):    "What do I contain?"
    V (Value):  "What information do I carry?"

    Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.d_k = embed_dim

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x:    (batch, seq_len, embed_dim)
        mask: (batch, seq_len, seq_len) — optional, e.g. causal mask
        """
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention scores: (B, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # Apply mask (e.g. causal: can't attend to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dim: each token's attention weights sum to 1
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Weighted sum of values
        out = attn_weights @ V   # (B, T, D)
        return out, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: run h attention heads in parallel,
    each learning to attend to different aspects.

    Example with 4 heads on "The cat sat on the mat":
    - Head 1 might learn syntactic relationships (subject-verb)
    - Head 2 might learn semantic similarity (cat-mat both concrete nouns)
    - Head 3 might learn positional proximity
    - Head 4 might learn coreference patterns
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (B, T, D) → (B, h, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention per head: (B, h, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention: (B, h, T, d_k)
        out = attn_weights @ V

        # Concatenate heads: (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), attn_weights


class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional encodings to give the model a sense of order.
    (Attention is permutation-invariant — it doesn't know which token came first)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    """
    One Transformer encoder block:
    x → LayerNorm → MultiHeadAttention → residual → LayerNorm → FFN → residual

    Key design choices:
    - Pre-LN (norm before attention) is more stable than Post-LN
    - Residual connections: gradient highway, prevents vanishing gradients
    - FFN expands dim by 4x then contracts: learns non-linear feature combos
    """
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


class TransformerClassifier(nn.Module):
    """
    Full Transformer for sequence classification.

    Architecture:
    Token embeddings + Positional encoding
    → N Transformer blocks
    → Global average pool over sequence
    → Linear classifier
    """
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers,
                 ff_dim, n_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, mask=None):
        tok_emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(tok_emb)

        all_attn = []
        for block in self.blocks:
            x, attn = block(x, mask)
            all_attn.append(attn)

        x = self.norm(x)
        # Global average pooling over sequence length → (B, D)
        x = x.mean(dim=1)
        return self.classifier(x), all_attn


# ── Part 2: Visualize Attention ───────────────────────────────────────────

def visualize_attention():
    """Show what attention weights look like on a toy sentence."""
    print("\n── Attention Visualization ──\n")

    words = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(words)
    embed_dim = 32

    # Create simple attention model
    attn = SelfAttention(embed_dim)

    # Random token embeddings
    x = torch.randn(1, seq_len, embed_dim)
    with torch.no_grad():
        _, weights = attn(x)

    weights_np = weights[0].detach().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(weights_np, xticklabels=words, yticklabels=words,
                annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Keys (attended to)")
    plt.ylabel("Queries (attending from)")
    plt.title("Self-Attention Weights\n(row i = how much token i attends to each token)")
    plt.tight_layout()
    plt.savefig("17_attention_weights.png", dpi=100)
    print("Saved attention weights to 17_attention_weights.png")


def visualize_positional_encoding():
    """Show the sinusoidal positional encoding patterns."""
    pe_module = PositionalEncoding(embed_dim=64, max_len=100)
    pe = pe_module.pe[0].numpy()  # (100, 64)

    plt.figure(figsize=(12, 5))
    plt.imshow(pe.T, aspect="auto", cmap="RdBu", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Position in sequence")
    plt.ylabel("Embedding dimension")
    plt.title("Positional Encoding — Sinusoidal Patterns\n"
              "(Each position gets a unique pattern of sin/cos waves)")
    plt.tight_layout()
    plt.savefig("17_positional_encoding.png", dpi=100)
    print("Saved positional encoding to 17_positional_encoding.png")


# ── Part 3: Train Transformer on Synthetic Task ───────────────────────────

def create_sequence_dataset(n=2000, seq_len=20, vocab_size=50):
    """
    Task: predict if sum of token IDs in sequence is > median (binary classification).
    Simple enough to verify Transformer learns something.
    """
    torch.manual_seed(42)
    X = torch.randint(1, vocab_size, (n, seq_len))
    sums = X.float().sum(dim=1)
    median = sums.median()
    y = (sums > median).long()
    return X, y


def train_transformer():
    print("\n── Training Transformer on Sequence Classification ──\n")

    VOCAB_SIZE  = 50
    EMBED_DIM   = 64
    N_HEADS     = 4
    N_LAYERS    = 2
    FF_DIM      = 128
    N_CLASSES   = 2
    SEQ_LEN     = 20
    BATCH_SIZE  = 64
    N_EPOCHS    = 20

    X, y = create_sequence_dataset(n=3000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds  = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM, n_classes=N_CLASSES,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    train_accs, test_accs = [], []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out, _ = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
        train_accs.append(correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out, _ = model(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total   += len(yb)
        test_accs.append(correct / total)
        scheduler.step()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} | Train: {train_accs[-1]:.1%} | Test: {test_accs[-1]:.1%}")

    # Training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_EPOCHS + 1), [a * 100 for a in train_accs], label="Train")
    plt.plot(range(1, N_EPOCHS + 1), [a * 100 for a in test_accs],  label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Transformer Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("17_transformer_training.png", dpi=100)
    print("\nSaved training curve to 17_transformer_training.png")
    return model


def main():
    print("=== Transformers & Attention from Scratch ===")
    print(f"Device: {DEVICE}\n")

    # ── Concepts ─────────────────────────────────────────────────────────
    print("── The Core Insight ──\n")
    print("RNNs process tokens sequentially: token3 must wait for token1, token2")
    print("Transformers process ALL tokens in parallel via attention")
    print("Attention lets every token directly see every other token\n")

    print("── Self-Attention Formula ──\n")
    print("  Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V\n")
    print("  Q = 'What am I looking for?'  (query)")
    print("  K = 'What do I contain?'      (key)")
    print("  V = 'What info do I carry?'   (value)")
    print("  sqrt(d_k) scaling prevents softmax from saturating\n")

    # ── Visualizations ───────────────────────────────────────────────────
    visualize_attention()
    visualize_positional_encoding()

    # ── Train ────────────────────────────────────────────────────────────
    model = train_transformer()

    # ── Save ─────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "17_transformer.pt")

    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Self-attention — Q, K, V projections + scaled dot-product")
    print("  ✓ Multi-head attention — parallel attention heads")
    print("  ✓ Positional encoding — sin/cos to inject sequence order")
    print("  ✓ Transformer block — attn + FFN + residuals + LayerNorm")
    print("  ✓ Full classifier — embeddings → blocks → pool → linear")
    print("  ✓ Gradient clipping — stabilizes Transformer training")
    print("\nThis is the exact architecture behind BERT, GPT, and all LLMs.")
    print("Next project: fine-tune BERT on real text data.")


if __name__ == "__main__":
    main()
