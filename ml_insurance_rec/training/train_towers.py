"""
Train the two-tower model with InfoNCE loss.

What happens here
─────────────────
1. Generate (or load) the product catalogue and synthetic user profiles.
2. Build a PyTorch Dataset of (user_features, positive_item_features) pairs
   drawn from the interaction log (clicked=1 events only — we want pairs where
   the user showed genuine interest, not random impressions).
3. Train with Adam + cosine-annealing LR schedule.
4. Evaluate recall@K on a held-out validation split after every epoch.
5. Embed all catalogue items with the trained item tower and write them to disk
   so the ANN index builder can read them.

Recall@K metric
───────────────
For each user in the validation set, retrieve the top-K items by cosine
similarity and check whether the true positive item appears in that list.
Recall@K = fraction of users for whom the positive appears in top-K.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from ..data.catalogue import generate_catalogue, save_catalogue, InsuranceProduct
from ..data.interaction_log import build_dataset, save_users, UserProfile
from ..models.two_tower import TwoTowerModel, EMBED_DIM

# ── Default hyper-parameters ──────────────────────────────────────────────────
DEFAULTS = dict(
    epochs        = 30,
    batch_size    = 256,
    lr            = 3e-4,
    val_fraction  = 0.1,
    seed          = 42,
    artifacts_dir = "artifacts",
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Each item is one (user_features, item_features) pair drawn from a
    clicked impression.  The two-tower model treats the item in each
    row as the positive; all other items in the same batch are negatives.
    """

    def __init__(
        self,
        users_by_id: dict[str, UserProfile],
        products_by_id: dict[str, InsuranceProduct],
        clicked_df,                    # DataFrame: user_id, product_id
    ):
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, row in clicked_df.iterrows():
            u = users_by_id.get(row["user_id"])
            p = products_by_id.get(row["product_id"])
            if u and p:
                self.pairs.append((u.to_feature_vector(), p.to_feature_vector()))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ufv, pfv = self.pairs[idx]
        return torch.from_numpy(ufv), torch.from_numpy(pfv)


# ── Recall@K evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_k(
    model: TwoTowerModel,
    val_loader: DataLoader,
    k: int = 20,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Compute Recall@K on the validation set using the in-batch approximation:
    for each batch, check whether the positive item is in the top-K retrieved
    from the batch's item pool.  This is a lower bound on true Recall@K
    (the real item pool is the full catalogue) but fast to compute during training.
    """
    model.eval()
    hits = total = 0
    for user_fv, item_fv in val_loader:
        user_fv, item_fv = user_fv.to(device), item_fv.to(device)
        user_emb, item_emb = model(user_fv, item_fv)

        # Similarity matrix within the batch
        sim    = torch.matmul(user_emb, item_emb.T)  # (B, B)
        B      = sim.size(0)
        top_k  = min(k, B)
        _, top_idx = sim.topk(top_k, dim=1)          # (B, k)

        diag   = torch.arange(B, device=device)       # correct index for each user
        hits  += (top_idx == diag.unsqueeze(1)).any(dim=1).sum().item()
        total += B

    model.train()
    return hits / max(total, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    epochs:        int   = DEFAULTS["epochs"],
    batch_size:    int   = DEFAULTS["batch_size"],
    lr:            float = DEFAULTS["lr"],
    val_fraction:  float = DEFAULTS["val_fraction"],
    seed:          int   = DEFAULTS["seed"],
    artifacts_dir: str   = DEFAULTS["artifacts_dir"],
) -> TwoTowerModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    art    = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Generating dataset …")
    users, products, interactions = build_dataset(seed=seed)

    save_catalogue(products, art / "catalogue.json")
    save_users(users, art / "users.json")
    interactions.to_parquet(art / "interactions.parquet", index=False)

    users_by_id    = {u.user_id: u for u in users}
    products_by_id = {p.id: p for p in products}

    clicked = interactions[interactions["clicked"] == 1]

    dataset   = InteractionDataset(users_by_id, products_by_id, clicked)
    n_val     = max(1, int(len(dataset) * val_fraction))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")

    # ── 2. Model, optimiser, scheduler ───────────────────────────────────────
    model     = TwoTowerModel().to(device)
    optimiser = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.01)

    best_recall = 0.0
    best_epoch  = 0

    # ── 3. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining two-tower model on {device} for {epochs} epochs …\n")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for user_fv, item_fv in train_loader:
            user_fv, item_fv = user_fv.to(device), item_fv.to(device)
            user_emb, item_emb = model(user_fv, item_fv)
            loss = model.infonce_loss(user_emb, item_emb)

            optimiser.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients early in training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        recall   = recall_at_k(model, val_loader, k=20, device=device)
        elapsed  = time.time() - t0
        tau      = model.temperature.item()

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={avg_loss:.4f}  recall@20={recall:.3f}  "
            f"τ={tau:.4f}  ({elapsed:.1f}s)"
        )

        if recall > best_recall:
            best_recall = recall
            best_epoch  = epoch
            model.save(art / "two_tower_best.pt")

    print(f"\nBest recall@20 = {best_recall:.3f} at epoch {best_epoch}")

    # ── 4. Load best checkpoint and embed the full catalogue ──────────────────
    model = TwoTowerModel.load(art / "two_tower_best.pt")
    model.to(device)
    model.eval()

    item_fvs = np.stack([p.to_feature_vector() for p in products])
    item_embs = model.embed_items(item_fvs)
    np.save(art / "item_embeddings.npy", item_embs)
    print(f"Item embeddings saved → {art / 'item_embeddings.npy'}  shape={item_embs.shape}")

    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    ap.add_argument("--batch_size",    type=int,   default=DEFAULTS["batch_size"])
    ap.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    ap.add_argument("--artifacts_dir", type=str,   default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    args = ap.parse_args()
    train(**vars(args))
