"""
Tests for the FastAPI serving layer.

We mock the loaded model so these tests are fast and don't require
a trained model artifact on disk.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


VALID_PAYLOAD = {
    "city": "London",
    "star_rating": 4,
    "amenities_count": 15,
    "distance_to_center_km": 2.5,
    "review_score": 8.7,
    "season": "peak",
    "rooms_available": 12,
}


@pytest.fixture(scope="module")
def client():
    """
    Create a test client with the model mocked at the module level.
    The mock returns a fixed prediction so tests are deterministic.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([175.50])

    # Patch the global model state inside main.py
    with patch("src.api.main._model_pipeline", mock_pipeline):
        from src.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestRootEndpoint:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_contains_service_info(self, client):
        body = client.get("/").json()
        assert "service" in body
        assert "docs" in body


class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_model_is_loaded(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["status"] == "healthy"

    def test_has_model_version(self, client):
        body = client.get("/health").json()
        assert "model_version" in body


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_has_required_fields(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "predicted_price_per_night" in body
        assert "confidence_interval" in body
        assert "model_version" in body
        assert "currency" in body

    def test_confidence_interval_is_ordered(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        ci = body["confidence_interval"]
        assert ci["lower"] < body["predicted_price_per_night"]
        assert ci["upper"] > body["predicted_price_per_night"]

    def test_predicted_price_is_positive(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["predicted_price_per_night"] > 0

    # ── Input validation ──────────────────────────────────────────────────────

    def test_invalid_star_rating_above_max(self, client):
        payload = {**VALID_PAYLOAD, "star_rating": 6}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_star_rating_below_min(self, client):
        payload = {**VALID_PAYLOAD, "star_rating": 2}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_city_rejected(self, client):
        payload = {**VALID_PAYLOAD, "city": "Atlantis"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_season_rejected(self, client):
        payload = {**VALID_PAYLOAD, "season": "monsoon"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_review_score_out_of_range(self, client):
        payload = {**VALID_PAYLOAD, "review_score": 11.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "city"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_all_cities_accepted(self, client):
        for city in ["New York", "London", "Paris", "Tokyo", "Dubai"]:
            payload = {**VALID_PAYLOAD, "city": city}
            assert client.post("/predict", json=payload).status_code == 200

    def test_all_seasons_accepted(self, client):
        for season in ["peak", "off-peak", "shoulder"]:
            payload = {**VALID_PAYLOAD, "season": season}
            assert client.post("/predict", json=payload).status_code == 200


class TestDegradedMode:
    def test_503_when_model_not_loaded(self):
        with patch("src.api.main._model_pipeline", None):
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/predict", json=VALID_PAYLOAD)
                assert resp.status_code == 503
