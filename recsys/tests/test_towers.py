"""Tests for the Two-Tower model and embedding components."""
import numpy as np
import pytest
import torch

from src.model.towers import (
    EmbeddingIndex,
    FeatureEncoder,
    TowerMLP,
    TwoTowerModel,
    train_two_tower,
)
from src.data_generator import generate_user_features, generate_hotel_features
import numpy as np


RNG = np.random.RandomState(42)
EMBEDDING_DIM = 16
HIDDEN_DIMS = [32, 16]


@pytest.fixture(scope="module")
def encoders_and_data():
    users = generate_user_features(200, RNG)
    hotels = generate_hotel_features(50, RNG)

    user_enc = FeatureEncoder()
    hotel_enc = FeatureEncoder()

    user_feats = user_enc.fit_transform(users.drop(columns=["user_id"]))
    hotel_feats = hotel_enc.fit_transform(hotels.drop(columns=["hotel_id"]))

    return user_enc, hotel_enc, user_feats, hotel_feats


class TestFeatureEncoder:
    def test_fit_transform_shape(self, encoders_and_data):
        user_enc, _, user_feats, _ = encoders_and_data
        assert user_feats.shape[0] == 200
        assert user_feats.shape[1] == user_enc.output_dim

    def test_no_nans_in_output(self, encoders_and_data):
        _, _, user_feats, hotel_feats = encoders_and_data
        assert not np.isnan(user_feats).any()
        assert not np.isnan(hotel_feats).any()

    def test_output_dim_matches_feature_count(self, encoders_and_data):
        user_enc, hotel_enc, _, _ = encoders_and_data
        assert user_enc.output_dim == len(user_enc.cat_cols) + len(user_enc.num_cols)


class TestTowerMLP:
    def test_output_shape(self, encoders_and_data):
        user_enc, _, user_feats, _ = encoders_and_data
        tower = TowerMLP(user_enc.output_dim, HIDDEN_DIMS, EMBEDDING_DIM)
        x = torch.tensor(user_feats[:10], dtype=torch.float32)
        out = tower(x)
        assert out.shape == (10, EMBEDDING_DIM)

    def test_output_is_l2_normalised(self, encoders_and_data):
        user_enc, _, user_feats, _ = encoders_and_data
        tower = TowerMLP(user_enc.output_dim, HIDDEN_DIMS, EMBEDDING_DIM)
        x = torch.tensor(user_feats[:20], dtype=torch.float32)
        out = tower(x)
        norms = out.norm(dim=1).detach().numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestTwoTowerModel:
    def test_forward_returns_scalar_per_pair(self, encoders_and_data):
        user_enc, hotel_enc, user_feats, hotel_feats = encoders_and_data
        model = TwoTowerModel(user_enc.output_dim, hotel_enc.output_dim, HIDDEN_DIMS, EMBEDDING_DIM)
        u = torch.tensor(user_feats[:8], dtype=torch.float32)
        h = torch.tensor(hotel_feats[:8], dtype=torch.float32)
        scores = model(u, h)
        assert scores.shape == (8,)

    def test_scores_in_valid_range(self, encoders_and_data):
        """Dot product of two unit vectors is in [-1, 1]."""
        user_enc, hotel_enc, user_feats, hotel_feats = encoders_and_data
        model = TwoTowerModel(user_enc.output_dim, hotel_enc.output_dim, HIDDEN_DIMS, EMBEDDING_DIM)
        u = torch.tensor(user_feats[:20], dtype=torch.float32)
        h = torch.tensor(hotel_feats[:20], dtype=torch.float32)
        scores = model(u, h).detach().numpy()
        assert (scores >= -1.01).all() and (scores <= 1.01).all()


class TestEmbeddingIndex:
    def test_top_k_returns_correct_count(self):
        embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
        # Normalise to unit vectors
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        hotel_ids = np.arange(100)
        index = EmbeddingIndex(embeddings, hotel_ids)
        user_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        user_emb /= np.linalg.norm(user_emb)
        top_k = index.get_top_k(user_emb, k=10)
        assert len(top_k) == 10

    def test_top_k_returns_valid_hotel_ids(self):
        embeddings = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        hotel_ids = np.arange(50)
        index = EmbeddingIndex(embeddings, hotel_ids)
        user_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        user_emb /= np.linalg.norm(user_emb)
        top_k = index.get_top_k(user_emb, k=5)
        assert all(hid in hotel_ids for hid in top_k)
