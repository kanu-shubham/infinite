"""
towers.py
---------
Two-Tower model for candidate retrieval.

Architecture
────────────

  User features ──► [User Tower MLP] ──► user_embedding (dim=32)
                                                  │
                                          dot product → relevance score
                                                  │
  Hotel features ─► [Hotel Tower MLP] ─► hotel_embedding (dim=32)

Both towers are simple feed-forward networks (Linear → ReLU → Linear → ReLU → Linear).
The output is an L2-normalised embedding vector.

Training objective: maximise the dot product for (user, hotel) pairs the user
liked (label=1) vs. random negative pairs (label=0).  This is pointwise binary
cross-entropy — simple and effective for a first production system.

In a real system at scale you would use:
  - In-batch negatives (faster, more GPU efficient)
  - Hard negative mining (more informative negatives)
  - ANN index (FAISS / ScaNN) for sub-millisecond retrieval at inference
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Feature encoders ──────────────────────────────────────────────────────────

class FeatureEncoder:
    """
    Converts raw user/hotel DataFrames into numeric tensors.

    Categorical columns are ordinal-encoded (simple integer mapping).
    Numerical columns are standardised (subtract mean, divide by std).

    Must be fit on training data, then saved alongside the model so
    inference uses the same encoding the model was trained on.
    """

    def __init__(self):
        self.cat_maps: dict[str, dict] = {}
        self.num_stats: dict[str, tuple[float, float]] = {}  # (mean, std)
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []

    def fit(self, df):
        import pandas as pd
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in self.cat_cols:
            unique = sorted(df[col].unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(unique)}

        for col in self.num_cols:
            self.num_stats[col] = (float(df[col].mean()), float(df[col].std()) + 1e-8)
        return self

    def transform(self, df) -> np.ndarray:
        parts = []
        for col in self.cat_cols:
            encoded = df[col].map(self.cat_maps[col]).fillna(0).astype(float).values
            parts.append(encoded.reshape(-1, 1))
        for col in self.num_cols:
            mean, std = self.num_stats[col]
            scaled = ((df[col].values - mean) / std).reshape(-1, 1)
            parts.append(scaled)
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def output_dim(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)


# ── Tower MLP ─────────────────────────────────────────────────────────────────

class TowerMLP(nn.Module):
    """
    A simple MLP that maps raw feature vectors to a normalised embedding.

    input_dim  → hidden_dims[0] → ReLU → hidden_dims[1] → ReLU → embedding_dim → L2 norm
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], embedding_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        # L2 normalise so dot product == cosine similarity ∈ [-1, 1]
        return F.normalize(emb, p=2, dim=-1)


# ── Two-Tower model ───────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Wraps both towers. Forward pass returns the dot-product similarity score.
    """

    def __init__(
        self,
        user_input_dim: int,
        hotel_input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
    ):
        super().__init__()
        self.user_tower = TowerMLP(user_input_dim, hidden_dims, embedding_dim)
        self.hotel_tower = TowerMLP(hotel_input_dim, hidden_dims, embedding_dim)

    def forward(
        self, user_features: torch.Tensor, hotel_features: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        hotel_emb = self.hotel_tower(hotel_features)
        # Dot product along embedding dimension → scalar score per pair
        return (user_emb * hotel_emb).sum(dim=1)

    def get_user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def get_hotel_embeddings(self, hotel_features: torch.Tensor) -> torch.Tensor:
        return self.hotel_tower(hotel_features)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_two_tower(
    model: TwoTowerModel,
    user_feats: np.ndarray,
    hotel_feats: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train the two-tower model with binary cross-entropy loss.

    Returns a list of per-epoch losses (for MLflow logging).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    model = model.to(device)

    X_user = torch.tensor(user_feats, dtype=torch.float32)
    X_hotel = torch.tensor(hotel_feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(X_user, X_hotel, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for u_batch, h_batch, y_batch in loader:
            u_batch, h_batch, y_batch = (
                u_batch.to(device),
                h_batch.to(device),
                y_batch.to(device),
            )
            optimiser.zero_grad()
            scores = model(u_batch, h_batch)
            loss = criterion(scores, y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(y_batch)

        avg_loss = total_loss / len(dataset)
        epoch_losses.append(avg_loss)
        logger.info(f"Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}")

    return epoch_losses


# ── Embedding index for retrieval ─────────────────────────────────────────────

class EmbeddingIndex:
    """
    Pre-computes and stores all hotel embeddings so retrieval at
    serving time is a single matrix multiply (O(n_hotels * embedding_dim)).

    In production with millions of hotels you would swap this for FAISS
    (Facebook AI Similarity Search) — exact same interface, ~100x faster.
    """

    def __init__(self, hotel_embeddings: np.ndarray, hotel_ids: np.ndarray):
        # shape: (n_hotels, embedding_dim)
        self.embeddings = hotel_embeddings.astype(np.float32)
        self.hotel_ids = hotel_ids

    def get_top_k(self, user_embedding: np.ndarray, k: int = 50) -> np.ndarray:
        """
        Returns the top-k hotel_ids most similar to the user embedding.
        user_embedding shape: (embedding_dim,)
        """
        # Dot products between user vector and all hotel vectors
        scores = self.embeddings @ user_embedding  # (n_hotels,)
        top_k_indices = np.argpartition(scores, -k)[-k:]
        # Sort the top-k by score descending
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        return self.hotel_ids[top_k_indices]
