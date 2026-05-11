"""End-to-end test: trains a tiny model in a temp dir, spins up the FastAPI app, hits /v1/rank."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml.models.ranker import Ranker
from ml.training.synthetic_data import generate


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    X, y = generate(4000, seed=7)
    split = int(0.85 * len(X))
    ranker = Ranker.train(X[:split], y[:split], X[split:], y[split:], num_boost_round=40)
    out = tmp_path_factory.mktemp("models") / "ranker.txt"
    ranker.save(out)
    return out


@pytest.fixture
def client(trained_model: Path):
    os.environ["RANKER_MODEL_PATH"] = str(trained_model)
    os.environ.pop("REDIS_URL", None)
    from ml.serving.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_rank_returns_slate(client):
    payload = {
        "request_id": "t1",
        "user": {"user_id": "u_1", "country": "US", "device": "mobile"},
        "context": {"destination": "PAR", "lead_time_days": 14, "los": 3, "pax": 2, "hour": 10, "dow": 3},
        "candidates": [
            {"ad_id": f"ad_{i}", "advertiser_id": f"adv_{i%10}", "bid_cpc": 0.5 + 0.1 * i}
            for i in range(20)
        ],
    }
    r = client.post("/v1/rank", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "t1"
    assert len(body["results"]) <= 8  # default slots
    slots = [x["slot"] for x in body["results"]]
    assert slots == sorted(slots)
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
    for r_ in body["results"]:
        assert r_["price_cpc"] >= 0.05


def test_rejects_too_many_candidates(client):
    payload = {
        "request_id": "t2",
        "user": {"user_id": "u_2", "country": "US", "device": "desktop"},
        "context": {"destination": "NYC", "lead_time_days": 1, "los": 1, "pax": 1, "hour": 0, "dow": 0},
        "candidates": [
            {"ad_id": f"ad_{i}", "advertiser_id": "adv_x", "bid_cpc": 1.0} for i in range(300)
        ],
    }
    r = client.post("/v1/rank", json=payload)
    assert r.status_code == 400


def test_health_and_ready(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
