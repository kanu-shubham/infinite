"""
Project 20: Cross-Attention & Encoder-Decoder Transformer
===========================================================
Cross-attention is how the decoder "reads" the encoder output.
It powers machine translation, image captioning, and every seq2seq model.

Difference from self-attention (Project 17):
  Self-attention:  Q, K, V all come from the SAME sequence
  Cross-attention: Q comes from the DECODER, K and V come from the ENCODER

What you'll learn:
- Cross-attention mechanism — Q from one sequence, K/V from another
- Full encoder-decoder Transformer architecture
- Causal (masked) self-attention — decoder can't peek at future tokens
- Sequence-to-sequence task: number sorting (easy to verify correctness)
- Greedy decoding and teacher forcing
- Visualizing cross-attention maps (what decoder looks at in the encoder)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens
PAD, BOS, EOS = 0, 1, 2   # padding, begin-of-sequence, end-of-sequence


# ── Attention building blocks ─────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Core attention function — shared by self-attention AND cross-attention.

    Q: (B, heads, T_q, d_k)   — queries
    K: (B, heads, T_k, d_k)   — keys
    V: (B, heads, T_k, d_v)   — values

    Key insight for cross-attention:
      T_q = decoder sequence length
      T_k = encoder sequence length  ← DIFFERENT from T_q
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, heads, T_q, T_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)   # each query attends over all keys
    out = weights @ V                      # (B, heads, T_q, d_v)
    return out, weights


class MultiHeadAttention(nn.Module):
    """
    Unified multi-head attention — works for both self-attention and cross-attention.
    The only difference is what you pass as query_input vs kv_input.

    Self-attention:   query_input = kv_input = x  (same sequence)
    Cross-attention:  query_input = decoder state, kv_input = encoder output
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query_input, kv_input, mask=None):
        """
        query_input: (B, T_q, D)  — where queries come from
        kv_input:    (B, T_k, D)  — where keys & values come from
                     For self-attention: kv_input == query_input
                     For cross-attention: kv_input is the encoder output
        """
        B = query_input.size(0)

        def project_and_split(linear, x):
            T = x.size(1)
            return linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = project_and_split(self.W_q, query_input)   # (B, h, T_q, d_k)
        K = project_and_split(self.W_k, kv_input)      # (B, h, T_k, d_k)
        V = project_and_split(self.W_v, kv_input)      # (B, h, T_k, d_k)

        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: (B, h, T_q, d_k) → (B, T_q, D)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        return self.W_o(out), attn_weights


# ── Encoder ───────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Encoder block: only self-attention (each token attends to every other token).
    No masking — encoder sees the full input sequence at once.
    """
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        # Self-attention: query and key/value both come from x
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), src_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, ff_dim,
                 max_len=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        B, T = src.shape
        positions = torch.arange(T, device=src.device).unsqueeze(0)
        x = self.dropout(self.embedding(src) + self.pos_embedding(positions))
        for block in self.blocks:
            x = block(x, src_mask)
        return self.norm(x)   # (B, T_src, embed_dim)


# ── Decoder ───────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Decoder block has THREE sub-layers:
    1. Masked self-attention  — decoder attends to its own past tokens (causal)
    2. CROSS-ATTENTION        — decoder queries the encoder output ← the key step
    3. Feed-forward network

    Cross-attention is what lets the decoder "read" the source sequence
    while generating each output token.
    """
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(embed_dim, n_heads)   # masked
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads)   # cross
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        # 1. Masked self-attention (can't look at future target tokens)
        sa_out, _ = self.self_attn(self.norm1(x), self.norm1(x), tgt_mask)
        x = x + sa_out

        # 2. CROSS-ATTENTION: Q from decoder, K and V from encoder output
        #    This is the bridge between encoder and decoder
        ca_out, cross_weights = self.cross_attn(
            query_input=self.norm2(x),           # decoder state
            kv_input=encoder_out,                # encoder output — DIFFERENT sequence!
            mask=src_mask,
        )
        x = x + ca_out

        # 3. Feed-forward
        x = x + self.ff(self.norm3(x))
        return x, cross_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, ff_dim,
                 max_len=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_out, tgt_mask=None, src_mask=None):
        B, T = tgt.shape
        positions = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.dropout(self.embedding(tgt) + self.pos_embedding(positions))

        all_cross_weights = []
        for block in self.blocks:
            x, cross_weights = block(x, encoder_out, tgt_mask, src_mask)
            all_cross_weights.append(cross_weights)

        return self.norm(x), all_cross_weights


# ── Full Encoder-Decoder Transformer ─────────────────────────────────────

class EncoderDecoderTransformer(nn.Module):
    """
    Complete encoder-decoder Transformer for sequence-to-sequence tasks.

    Used in: machine translation, summarization, code generation,
             speech recognition, image captioning (with visual encoder).
    """
    def __init__(self, src_vocab, tgt_vocab, embed_dim=64,
                 n_heads=4, n_layers=2, ff_dim=128, max_len=50, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, n_heads, n_layers, ff_dim, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, embed_dim, n_heads, n_layers, ff_dim, max_len, dropout)
        self.output_proj = nn.Linear(embed_dim, tgt_vocab)

    def make_causal_mask(self, T, device):
        """Lower-triangular mask: position i can only attend to positions <= i."""
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    def make_pad_mask(self, seq, pad_idx=PAD):
        """Mask out padding tokens."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_len  = tgt.size(1)
        tgt_mask = self.make_causal_mask(tgt_len, tgt.device) & \
                   self.make_pad_mask(tgt)

        encoder_out = self.encoder(src, src_mask)
        decoder_out, cross_weights = self.decoder(tgt, encoder_out, tgt_mask, src_mask)

        logits = self.output_proj(decoder_out)  # (B, T_tgt, tgt_vocab)
        return logits, cross_weights

    @torch.no_grad()
    def greedy_decode(self, src, max_len=20):
        """Auto-regressively generate output one token at a time."""
        self.eval()
        src = src.to(DEVICE)
        src_mask = self.make_pad_mask(src)
        encoder_out = self.encoder(src, src_mask)

        # Start with BOS token
        tgt = torch.tensor([[BOS]], device=DEVICE)

        all_cross_attn = []
        for _ in range(max_len):
            tgt_len  = tgt.size(1)
            tgt_mask = self.make_causal_mask(tgt_len, DEVICE)
            dec_out, cross_w = self.decoder(tgt, encoder_out, tgt_mask, src_mask)
            all_cross_attn.append(cross_w[-1][:, 0, -1:, :])  # last layer, head 0, last token

            next_token = self.output_proj(dec_out[:, -1:]).argmax(-1)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == EOS:
                break

        return tgt[0, 1:], torch.cat(all_cross_attn, dim=2)  # strip BOS


# ── Seq2Seq Task: Number Sorting ──────────────────────────────────────────

class SortingDataset(Dataset):
    """
    Task: given a shuffled sequence of numbers, output the sorted sequence.
    Input:  [3, 1, 4, 1, 5] + BOS/EOS
    Output: [1, 1, 3, 4, 5] + BOS/EOS

    Simple enough to verify the model learned cross-attention correctly.
    """
    def __init__(self, n=5000, seq_len=8, vocab_size=12):
        # Tokens 3+ are numbers (0=PAD, 1=BOS, 2=EOS, 3..N are values)
        self.data = []
        offset = 3  # shift numbers to avoid special tokens
        for _ in range(n):
            nums   = torch.randint(0, vocab_size - offset, (seq_len,)) + offset
            sorted_nums = nums.sort().values

            src = torch.cat([torch.tensor([BOS]), nums,        torch.tensor([EOS])])
            tgt = torch.cat([torch.tensor([BOS]), sorted_nums, torch.tensor([EOS])])
            self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_seq2seq():
    print("\n── Training Encoder-Decoder on Number Sorting ──\n")

    VOCAB      = 15    # PAD, BOS, EOS + numbers 3-14
    EMBED_DIM  = 64
    N_HEADS    = 4
    N_LAYERS   = 2
    FF_DIM     = 128
    N_EPOCHS   = 30
    BATCH_SIZE = 128

    dataset = SortingDataset(n=8000, seq_len=6, vocab_size=VOCAB)
    split   = int(0.85 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = EncoderDecoderTransformer(
        src_vocab=VOCAB, tgt_vocab=VOCAB,
        embed_dim=EMBED_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    train_losses, test_accs = [], []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            # Teacher forcing: feed ground-truth target as decoder input
            # (except the last token — we predict that)
            dec_input  = tgt[:, :-1]  # [BOS, t1, t2, ...]
            dec_target = tgt[:, 1:]   # [t1, t2, ..., EOS]

            optimizer.zero_grad()
            logits, _ = model(src, dec_input)
            loss = criterion(logits.reshape(-1, VOCAB), dec_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate: full sequence accuracy
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            correct_seq, total_seq = 0, 0
            with torch.no_grad():
                for src, tgt in test_loader:
                    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                    dec_input  = tgt[:, :-1]
                    dec_target = tgt[:, 1:]
                    logits, _ = model(src, dec_input)
                    preds = logits.argmax(-1)
                    correct_seq += (preds == dec_target).all(dim=1).sum().item()
                    total_seq   += len(src)
            seq_acc = correct_seq / total_seq
            test_accs.append(seq_acc)
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} | Loss: {train_losses[-1]:.4f} | Seq Acc: {seq_acc:.1%}")

    return model, train_losses, test_accs, VOCAB


# ── Visualize cross-attention ─────────────────────────────────────────────

def visualize_cross_attention(model, vocab_size):
    """
    Show what each decoder output token attends to in the encoder input.
    This is the most intuitive way to understand cross-attention.
    """
    print("\n── Cross-Attention Visualization ──\n")

    offset = 3
    # Input: shuffled sequence [5, 3, 7, 4, 6, 3]
    nums = torch.tensor([5, 3, 7, 4, 6, 3]) + offset
    src  = torch.cat([torch.tensor([BOS]), nums, torch.tensor([EOS])]).unsqueeze(0)

    sorted_output, cross_attn = model.greedy_decode(src, max_len=10)
    src_tokens = src[0].tolist()

    print("Input:  ", [t - offset if t >= offset else {PAD:"PAD",BOS:"BOS",EOS:"EOS"}[t] for t in src_tokens])
    print("Output: ", [t.item() - offset if t.item() >= offset else {PAD:"PAD",BOS:"BOS",EOS:"EOS"}[t.item()] for t in sorted_output])

    # cross_attn: (1, 1, T_out, T_in)
    attn_map = cross_attn[0, 0].cpu().numpy()

    src_labels = []
    for t in src_tokens:
        if t == BOS:   src_labels.append("BOS")
        elif t == EOS: src_labels.append("EOS")
        else:          src_labels.append(str(t - offset))

    tgt_labels = []
    for t in sorted_output:
        tv = t.item()
        if tv == EOS: tgt_labels.append("EOS")
        else:         tgt_labels.append(str(tv - offset))

    if len(tgt_labels) > 0 and attn_map.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        n_out = min(attn_map.shape[0], len(tgt_labels))
        n_in  = min(attn_map.shape[1], len(src_labels))
        sns.heatmap(
            attn_map[:n_out, :n_in],
            xticklabels=src_labels[:n_in],
            yticklabels=tgt_labels[:n_out],
            cmap="Blues", annot=True, fmt=".2f",
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Encoder input tokens (source)")
        ax.set_ylabel("Decoder output tokens (target)")
        ax.set_title(
            "Cross-Attention Map\n"
            "Row i = what decoder position i attended to in the encoder\n"
            "(bright = high attention)"
        )
        plt.tight_layout()
        plt.savefig("20_cross_attention_map.png", dpi=100)
        print("\nSaved cross-attention map to 20_cross_attention_map.png")


def compare_self_vs_cross():
    """Side-by-side diagram of self-attention vs cross-attention."""
    print("\n── Self-Attention vs Cross-Attention ──\n")
    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │              SELF-ATTENTION (Encoder / Project 17)          │
  │                                                             │
  │   "The cat sat on the mat"                                  │
  │      ↓    ↓   ↓   ↓   ↓   ↓                                │
  │   [  Q    K   V  from the SAME sequence  ]                  │
  │      ↓                                                      │
  │   Each token attends to every other token in itself         │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │            CROSS-ATTENTION (Decoder — this project)         │
  │                                                             │
  │  Encoder output: "Le chat est assis sur le tapis"           │
  │                   ↓    ↓    ↓    ↓    ↓    ↓    ↓           │
  │                  [      K,  V  from ENCODER      ]          │
  │                                        ↑                    │
  │  Decoder state:  "The cat ___"  →  [ Q from DECODER ]       │
  │                                        ↓                    │
  │  Decoder asks: "To generate 'sat', which encoder            │
  │                 tokens should I look at?"                   │
  │  Answer:         "est" + "assis" (highest attention)        │
  └─────────────────────────────────────────────────────────────┘

  Used in:
    • Machine translation (original Transformer, "Attention Is All You Need")
    • Summarization (encoder=article, decoder=summary)
    • Image captioning (encoder=CNN features, decoder=text)
    • Speech recognition (encoder=audio, decoder=text)
    • DALL-E / Stable Diffusion (encoder=text prompt, decoder=image)
    """)


def main():
    print("=== Cross-Attention & Encoder-Decoder Transformer ===")
    print(f"Device: {DEVICE}\n")

    compare_self_vs_cross()

    # Train
    model, train_losses, test_accs, vocab_size = train_seq2seq()

    # Visualize cross-attention
    visualize_cross_attention(model, vocab_size)

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(train_losses, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    eval_epochs = [1] + list(range(5, len(train_losses) + 1, 5))
    axes[1].plot(eval_epochs[:len(test_accs)], [a * 100 for a in test_accs], "o-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sequence Accuracy (%)")
    axes[1].set_title("Full-Sequence Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Encoder-Decoder Transformer — Number Sorting", fontsize=13)
    plt.tight_layout()
    plt.savefig("20_encoder_decoder_training.png", dpi=100)
    print("Saved training curves to 20_encoder_decoder_training.png")

    # Save
    torch.save(model.state_dict(), "20_seq2seq_transformer.pt")

    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Cross-attention — Q from decoder, K and V from encoder")
    print("  ✓ Encoder — bidirectional self-attention over full source")
    print("  ✓ Decoder — 3 sub-layers: masked self-attn + cross-attn + FFN")
    print("  ✓ Causal mask — decoder can't peek at future target tokens")
    print("  ✓ Teacher forcing — feed ground truth during training")
    print("  ✓ Greedy decoding — generate one token at a time at inference")
    print("  ✓ Cross-attention map — visualize what decoder looks at")
    print("\nThis architecture is the backbone of:")
    print("  - Original Transformer ('Attention Is All You Need', 2017)")
    print("  - T5, BART, mT5 (encoder-decoder language models)")
    print("  - Whisper (speech recognition)")
    print("  - DALL-E 1 (text-to-image)")


if __name__ == "__main__":
    main()
