"""
Two-Tower model — trained end-to-end with InfoNCE (in-batch negatives).

Architecture
────────────
User Tower:  user_features (53) → Linear(256) → BN → ReLU → Dropout(0.2)
                                → Linear(128) → ReLU → Linear(embed_dim=128)
                                → L2-normalise

Item Tower:  item_features (50) → Linear(256) → BN → ReLU → Dropout(0.2)
                                → Linear(128) → ReLU → Linear(embed_dim=128)
                                → L2-normalise

Loss: InfoNCE with in-batch negatives and learnable temperature τ.
  For a batch of B (user, positive_item) pairs:
    similarity matrix  S[i,j] = user_emb[i] · item_emb[j] / τ
    loss = 0.5 * (CE(S, eye) + CE(Sᵀ, eye))
  All B-1 other items in the batch serve as negatives for each anchor.

After training:
  • All item embeddings are computed offline and loaded into the ANN index.
  • At serving time only the user tower runs on the incoming request.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.catalogue import ITEM_FEATURE_DIM
from ..data.interaction_log import USER_FEATURE_DIM

EMBED_DIM = 128


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim),
    )


class UserTower(nn.Module):
    def __init__(self, in_dim: int = USER_FEATURE_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = _mlp(in_dim, 256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unit-normalised user embeddings, shape (B, embed_dim)."""
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, in_dim: int = ITEM_FEATURE_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = _mlp(in_dim, 256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unit-normalised item embeddings, shape (B, embed_dim)."""
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    """
    Joint two-tower retrieval model.

    Parameters
    ----------
    user_in_dim : int
        Dimensionality of the user feature vector (default 53).
    item_in_dim : int
        Dimensionality of the item feature vector (default 50).
    embed_dim : int
        Output embedding dimension shared by both towers (default 128).
    init_temperature : float
        Starting value for the learnable softmax temperature τ.
        Smaller τ → sharper softmax → harder negatives.
    """

    def __init__(
        self,
        user_in_dim: int = USER_FEATURE_DIM,
        item_in_dim: int = ITEM_FEATURE_DIM,
        embed_dim: int   = EMBED_DIM,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(user_in_dim, embed_dim)
        self.item_tower = ItemTower(item_in_dim, embed_dim)
        # Log-temperature is learnable — keeps τ positive via exp()
        self.log_temp   = nn.Parameter(torch.tensor(np.log(init_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        user_features : (B, user_in_dim)
        item_features : (B, item_in_dim)  — one positive item per user

        Returns
        -------
        user_emb : (B, embed_dim)  unit vectors
        item_emb : (B, embed_dim)  unit vectors
        """
        return self.user_tower(user_features), self.item_tower(item_features)

    def infonce_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE loss with in-batch negatives.

        The diagonal of the similarity matrix is the positive pair;
        all off-diagonal entries are treated as negatives.

        Loss = 0.5 * (CE(S, labels) + CE(Sᵀ, labels))
        where S[i,j] = (user_emb[i] · item_emb[j]) / τ
        """
        B      = user_emb.size(0)
        S      = torch.matmul(user_emb, item_emb.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=S.device)
        loss   = 0.5 * (F.cross_entropy(S, labels) + F.cross_entropy(S.T, labels))
        return loss

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "TwoTowerModel":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    # ── Convenience inference methods ─────────────────────────────────────────

    @torch.no_grad()
    def embed_users(self, user_features: np.ndarray) -> np.ndarray:
        """Batch-encode user feature matrix → numpy float32 (N, embed_dim)."""
        x   = torch.from_numpy(user_features).float()
        emb = self.user_tower(x)
        return emb.cpu().numpy()

    @torch.no_grad()
    def embed_items(self, item_features: np.ndarray) -> np.ndarray:
        """Batch-encode item feature matrix → numpy float32 (N, embed_dim)."""
        x   = torch.from_numpy(item_features).float()
        emb = self.item_tower(x)
        return emb.cpu().numpy()

    @torch.no_grad()
    def embed_single_user(self, user_fv: np.ndarray) -> np.ndarray:
        """Encode a single user feature vector (53,) → (embed_dim,)."""
        x   = torch.from_numpy(user_fv[None]).float()          # (1, 53)
        emb = self.user_tower(x)
        return emb.squeeze(0).cpu().numpy()
