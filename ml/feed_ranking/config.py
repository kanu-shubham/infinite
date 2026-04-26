"""Hyperparameters and feature/vocab dimensions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

POST_TYPES: tuple[str, ...] = ("text", "image", "video", "article", "job")
TOPICS: tuple[str, ...] = (
    "engineering", "ml", "design", "product", "marketing",
    "finance", "hr", "sales", "leadership", "career",
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CHECKPOINT = ARTIFACT_DIR / "model.pt"
DEFAULT_DATASET = ARTIFACT_DIR / "events.json"


@dataclass
class FeatureSpec:
    """Schema of the (user, post, context) feature vector."""

    numeric: tuple[str, ...] = (
        "user_age_days",
        "user_num_connections",
        "user_num_followers",
        "post_age_hours",
        "post_num_likes",
        "post_num_comments",
        "post_num_shares",
        "post_text_length",
        "author_num_followers",
        "is_connected",
        "topic_affinity",
        "hour_of_day",
    )
    post_type_vocab: tuple[str, ...] = POST_TYPES
    topic_vocab: tuple[str, ...] = TOPICS

    @property
    def dim(self) -> int:
        return len(self.numeric) + len(self.post_type_vocab) + len(self.topic_vocab)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    negative_downsample_ratio: float = 4.0  # 4 negatives per positive in train
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 17
    device: str = "cpu"
    model_kind: str = "mlp"  # "mlp" | "logreg"
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
