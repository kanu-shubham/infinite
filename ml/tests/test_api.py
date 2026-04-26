"""Integration test of the FastAPI service."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Re-route artefacts to a temp dir so the test is hermetic.
    import ml.feed_ranking.config as config

    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_CHECKPOINT", tmp_path / "model.pt")
    monkeypatch.setattr(config, "DEFAULT_DATASET", tmp_path / "events.json")

    # Reload api so it picks up the patched paths.
    import ml.api.main as api_main

    importlib.reload(api_main)
    # Reset the in-process store between runs.
    from ml.feed_ranking import store as store_mod

    store_mod.set_store(store_mod.Store())
    return TestClient(api_main.app)


def _seed(client: TestClient) -> None:
    from ml.feed_ranking.data import GenerationConfig, generate_dataset
    from ml.feed_ranking.store import get_store

    generate_dataset(
        GenerationConfig(n_users=60, n_posts=200, n_impressions=4_000, seed=21),
        store=get_store(),
    )


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_feed_requires_trained_model(client):
    r = client.get("/v1/feed/u_0")
    # 404 (unknown user) before seeding, 503 (no model) after seeding.
    assert r.status_code in (404, 503)


def test_full_flow_train_then_rank(client):
    _seed(client)

    train_r = client.post("/v1/model/train", json={"epochs": 3, "model_kind": "mlp"})
    assert train_r.status_code == 200, train_r.text
    body = train_r.json()
    assert body["n_train"] > 0
    assert 0.0 <= body["val_auc"] <= 1.0

    info = client.get("/v1/model/info").json()
    assert info["loaded"] is True
    assert info["model_kind"] == "mlp"

    feed_r = client.get("/v1/feed/u_0?k=10")
    assert feed_r.status_code == 200
    ranked = feed_r.json()["ranked"]
    assert len(ranked) == 10
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)

    pred_r = client.post(
        "/v1/model/predict",
        json={"items": [{"user_id": "u_0", "post_id": ranked[0]["post_id"]}]},
    )
    assert pred_r.status_code == 200
    assert len(pred_r.json()["scores"]) == 1


def test_ingest_event_validation(client):
    _seed(client)
    # unknown user
    r = client.post(
        "/v1/events/click",
        json={"user_id": "ghost", "post_id": "p_0"},
    )
    assert r.status_code == 404
