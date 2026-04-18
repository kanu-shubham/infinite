"""
Project 24: Recommender Systems
=================================
Build the systems that power Netflix, Spotify, Amazon, and YouTube.

Two major approaches:
1. Matrix Factorization — collaborative filtering via latent factors
2. Two-Tower Model    — the modern neural approach used at Google/Meta scale

What you'll learn:
- Collaborative filtering: "users like you also liked..."
- Matrix factorization: decompose ratings into user & item embeddings
- Cold-start problem: what happens with new users/items
- Two-tower model: separate encoders for users and items
- Evaluation: RMSE, Precision@K, Recall@K, NDCG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic dataset ─────────────────────────────────────────────────────

def create_ratings_dataset(n_users=500, n_items=200, n_ratings=20000,
                            n_user_factors=5, n_item_factors=5):
    """
    Generate synthetic ratings that follow a low-rank structure
    (the assumption behind matrix factorization).
    """
    np.random.seed(42)

    # True latent factors (hidden preferences)
    user_factors = np.random.randn(n_users, n_user_factors)
    item_factors = np.random.randn(n_items, n_item_factors)
    user_bias    = np.random.randn(n_users) * 0.5
    item_bias    = np.random.randn(n_items) * 0.5
    global_mean  = 3.5

    rows = np.random.randint(0, n_users, n_ratings)
    cols = np.random.randint(0, n_items, n_ratings)

    true_ratings = (
        global_mean
        + user_bias[rows]
        + item_bias[cols]
        + (user_factors[rows] * item_factors[cols]).sum(axis=1)
    )
    noisy_ratings = np.clip(true_ratings + np.random.randn(n_ratings) * 0.5, 1, 5)
    noisy_ratings = np.round(noisy_ratings * 2) / 2   # round to 0.5 steps

    df = pd.DataFrame({"user_id": rows, "item_id": cols, "rating": noisy_ratings})
    df = df.drop_duplicates(subset=["user_id", "item_id"])

    print(f"Users: {n_users}, Items: {n_items}, Ratings: {len(df)}")
    print(f"Sparsity: {1 - len(df)/(n_users*n_items):.1%} (most user-item pairs unrated)")
    print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}\n")

    return df, n_users, n_items


# ── Model 1: Matrix Factorization ─────────────────────────────────────────

class MatrixFactorization(nn.Module):
    """
    Classic collaborative filtering via matrix factorization.

    Rating(u, i) ≈ μ + b_u + b_i + U_u · I_i

    Where:
      μ   = global mean rating
      b_u = user bias (some users always rate high/low)
      b_i = item bias (some items are universally loved/hated)
      U_u = user embedding (latent taste vector)
      I_i = item embedding (latent feature vector)
      U_u · I_i = dot product = how well user u's taste matches item i

    This is the same algorithm that won the Netflix Prize ($1M, 2009).
    """
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, n_factors)
        self.item_emb  = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_mean = nn.Parameter(torch.tensor(3.5))

        # Initialize embeddings
        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.item_emb.weight,  std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        U  = self.user_emb(user_ids)   # (B, n_factors)
        I  = self.item_emb(item_ids)   # (B, n_factors)
        bu = self.user_bias(user_ids).squeeze(1)
        bi = self.item_bias(item_ids).squeeze(1)

        dot = (U * I).sum(dim=1)       # element-wise product then sum
        return self.global_mean + bu + bi + dot


# ── Model 2: Neural Collaborative Filtering (NCF) ────────────────────────

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering: replace dot product with an MLP.
    Learns non-linear user-item interactions.
    """
    def __init__(self, n_users, n_items, n_factors=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        self.mlp = nn.Sequential(
            nn.Linear(n_factors * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        U = self.user_emb(user_ids)
        I = self.item_emb(item_ids)
        x = torch.cat([U, I], dim=1)
        return self.mlp(x).squeeze(1) + 3.5


# ── Model 3: Two-Tower Model ──────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Two-Tower (Dual Encoder) Model — used at Google, Meta, YouTube scale.

    Architecture:
      User Tower:  user features → user embedding vector
      Item Tower:  item features → item embedding vector
      Score:       cosine similarity between the two towers

    Key advantage over MF:
    - Can incorporate rich features (age, genre, text, images)
    - Item tower can be pre-computed and indexed for fast retrieval
    - Powers YouTube recommendations, Pinterest search, Google Play

    At inference: embed all items once → find nearest neighbors to user vector.
    """
    def __init__(self, n_users, n_items, n_user_features, n_item_features,
                 embedding_dim=64):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, 32)
        self.item_id_emb = nn.Embedding(n_items, 32)

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(32 + n_user_features, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(32 + n_item_features, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode_user(self, user_ids, user_features):
        id_emb = self.user_id_emb(user_ids)
        x = torch.cat([id_emb, user_features], dim=1)
        return self.user_tower(x)

    def encode_item(self, item_ids, item_features):
        id_emb = self.item_id_emb(item_ids)
        x = torch.cat([id_emb, item_features], dim=1)
        return self.item_tower(x)

    def forward(self, user_ids, user_features, item_ids, item_features):
        user_vec = self.encode_user(user_ids, user_features)
        item_vec = self.encode_item(item_ids, item_features)
        # Cosine similarity scaled to rating range
        cos_sim = torch.nn.functional.cosine_similarity(user_vec, item_vec)
        return cos_sim * self.scale * 2 + 3.5


# ── Dataset & Training ────────────────────────────────────────────────────

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users   = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items   = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,  dtype=torch.float32)

    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.ratings[idx]


def train_model(model, train_loader, test_loader, n_epochs=20, lr=1e-3, wd=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    history = {"train_rmse": [], "test_rmse": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            optimizer.zero_grad()
            preds = model(users, items)
            loss  = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            all_preds, all_true = [], []
            for users, items, ratings in test_loader:
                users, items = users.to(DEVICE), items.to(DEVICE)
                preds = model(users, items).cpu()
                all_preds.extend(preds.numpy())
                all_true.extend(ratings.numpy())

        train_rmse = np.sqrt(np.mean(losses))
        test_rmse  = np.sqrt(mean_squared_error(all_true, all_preds))
        history["train_rmse"].append(train_rmse)
        history["test_rmse"].append(test_rmse)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    return history


def precision_recall_at_k(model, df, n_users, n_items, k=10, threshold=4.0):
    """Evaluate ranking quality: how well do top-K recommendations match true preferences?"""
    model.eval()
    precisions, recalls = [], []

    sample_users = np.random.choice(n_users, size=50, replace=False)

    with torch.no_grad():
        for u in sample_users:
            user_ratings = df[df["user_id"] == u]
            if len(user_ratings) < 5:
                continue

            # True positives: items user actually rated highly
            true_positives = set(user_ratings[user_ratings["rating"] >= threshold]["item_id"].values)
            if not true_positives:
                continue

            # Score all items
            all_items = torch.arange(n_items).to(DEVICE)
            all_users = torch.full((n_items,), u, dtype=torch.long).to(DEVICE)
            scores    = model(all_users, all_items).cpu().numpy()

            # Top-K recommended items
            top_k = set(np.argsort(scores)[-k:])

            hits = len(top_k & true_positives)
            precisions.append(hits / k)
            recalls.append(hits / max(len(true_positives), 1))

    return np.mean(precisions), np.mean(recalls)


def main():
    print("=== Recommender Systems ===")
    print(f"Device: {DEVICE}\n")

    df, n_users, n_items = create_ratings_dataset()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = RatingsDataset(train_df)
    test_ds  = RatingsDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=512)

    # ── Train all models ──────────────────────────────────────────────────
    models = {
        "Matrix Factorization": MatrixFactorization(n_users, n_items, n_factors=20).to(DEVICE),
        "Neural CF":            NeuralCF(n_users, n_items, n_factors=32).to(DEVICE),
    }

    histories = {}
    print("=== Training Models ===\n")

    for name, model in models.items():
        print(f"── {name} ──")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        h = train_model(model, train_loader, test_loader, n_epochs=20)
        histories[name] = h
        p_at_k, r_at_k = precision_recall_at_k(model, df, n_users, n_items, k=10)
        print(f"   Precision@10: {p_at_k:.3f}  |  Recall@10: {r_at_k:.3f}\n")

    # ── Visualize training ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, h in histories.items():
        axes[0].plot(h["train_rmse"], label=f"{name} (train)")
        axes[1].plot(h["test_rmse"],  label=f"{name} (test)")

    for ax, title in [(axes[0], "Training RMSE"), (axes[1], "Test RMSE")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Recommender Systems: Training Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("24_recommender_training.png", dpi=100)
    print("Saved training curves to 24_recommender_training.png")

    # ── Visualize embeddings ──────────────────────────────────────────────
    mf = models["Matrix Factorization"]
    user_embs = mf.user_emb.weight.detach().cpu().numpy()
    item_embs = mf.item_emb.weight.detach().cpu().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    all_embs = pca.fit_transform(np.vstack([user_embs, item_embs]))
    u_2d = all_embs[:n_users]
    i_2d = all_embs[n_users:]

    plt.figure(figsize=(10, 7))
    plt.scatter(u_2d[:, 0], u_2d[:, 1], alpha=0.3, s=15, label="Users", c="blue")
    plt.scatter(i_2d[:, 0], i_2d[:, 1], alpha=0.3, s=15, label="Items", c="red")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Learned User & Item Embeddings (PCA projection)\n"
              "Nearby users/items share similar tastes/features")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("24_recommender_embeddings.png", dpi=100)
    print("Saved embedding plot to 24_recommender_embeddings.png")

    # ── Two-Tower architecture explanation ────────────────────────────────
    print("\n── Two-Tower Model (Production Architecture) ──\n")
    print("""
  Used by: YouTube (2016+), Google Play, Pinterest, Airbnb

  User Tower            Item Tower
  ───────────           ───────────
  User ID emb           Item ID emb
  Age, location         Genre, price
  Watch history         Avg rating
  ↓                     ↓
  MLP                   MLP
  ↓                     ↓
  [user vector] ←── cosine similarity ──→ [item vector]

  At training:  optimize similarity for interacted pairs
  At inference: embed all items once → store in vector DB
                for each user → find nearest item vectors
                (approximate nearest neighbor search: FAISS, ScaNN)

  This scales to billions of items because item embeddings
  are pre-computed — only the user tower runs at serving time.
    """)

    print("=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Collaborative filtering — 'users like you also liked'")
    print("  ✓ Matrix factorization — user/item latent factors + biases")
    print("  ✓ Neural CF — MLP replaces dot product for non-linear interactions")
    print("  ✓ Two-tower model — scalable dual-encoder architecture")
    print("  ✓ Precision@K / Recall@K — ranking quality metrics")
    print("  ✓ Embedding visualization — PCA on learned user/item spaces")


if __name__ == "__main__":
    main()
