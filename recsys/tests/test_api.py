"""Tests for the recommendation API endpoints."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


VALID_USER = {
    "age_group": "26-35",
    "travel_type": "business",
    "price_sensitivity": 0.3,
    "review_weight": 0.8,
    "prefers_city_center": 1,
    "loyalty_tier": "gold",
}


def _make_mock_models():
    """Return a consistent set of mocked model objects."""
    import torch
    import pandas as pd
    from src.data_generator import generate_hotel_features
    import numpy as np

    rng = np.random.RandomState(0)
    hotel_df = generate_hotel_features(50, rng)

    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.zeros((1, 8), dtype=np.float32)
    mock_encoder.cat_cols = ["city", "price_tier"]
    mock_encoder.num_cols = ["star_rating", "avg_review_score",
                              "distance_to_center_km", "amenities_count",
                              "has_pool", "has_spa"]

    mock_tower = MagicMock()
    mock_tower.get_user_embedding.return_value = torch.zeros(1, 16)

    mock_index = MagicMock()
    mock_index.get_top_k.return_value = np.arange(10)

    mock_ranker = MagicMock()
    mock_ranker.rank_candidates.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return mock_tower, mock_encoder, mock_index, mock_ranker, hotel_df


@pytest.fixture(scope="module")
def client():
    mock_tower, mock_enc, mock_index, mock_ranker, hotel_df = _make_mock_models()
    patches = {
        "src.api.main._tower_model": mock_tower,
        "src.api.main._user_encoder": mock_enc,
        "src.api.main._hotel_encoder": mock_enc,
        "src.api.main._embedding_index": mock_index,
        "src.api.main._ranker": mock_ranker,
        "src.api.main._hotel_df": hotel_df,
    }
    with patch.multiple("src.api.main", **{k.split(".")[-1]: v for k, v in patches.items()}):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


class TestRootEndpoint:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_shows_architecture(self, client):
        body = client.get("/").json()
        assert "Two-Tower" in body.get("architecture", "")


class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200


class TestRecommendEndpoint:
    def test_valid_request_returns_200(self, client):
        assert client.post("/recommend", json=VALID_USER).status_code == 200

    def test_response_has_recommendations(self, client):
        body = client.post("/recommend", json=VALID_USER).json()
        assert "recommendations" in body
        assert len(body["recommendations"]) > 0

    def test_recommendations_are_ranked(self, client):
        body = client.post("/recommend", json=VALID_USER).json()
        ranks = [r["rank"] for r in body["recommendations"]]
        assert ranks == sorted(ranks)

    def test_all_travel_types_accepted(self, client):
        for tt in ["business", "leisure", "family", "solo"]:
            payload = {**VALID_USER, "travel_type": tt}
            assert client.post("/recommend", json=payload).status_code == 200

    def test_invalid_travel_type_rejected(self, client):
        payload = {**VALID_USER, "travel_type": "spaceship"}
        assert client.post("/recommend", json=payload).status_code == 422

    def test_price_sensitivity_out_of_range(self, client):
        payload = {**VALID_USER, "price_sensitivity": 1.5}
        assert client.post("/recommend", json=payload).status_code == 422

    def test_missing_field_rejected(self, client):
        payload = {k: v for k, v in VALID_USER.items() if k != "loyalty_tier"}
        assert client.post("/recommend", json=payload).status_code == 422
