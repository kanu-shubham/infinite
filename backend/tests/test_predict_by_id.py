"""The entity-based serving path, end to end through HTTP."""

import pytest

from app.ml import feature_store, model_cache


@pytest.fixture(autouse=True)
def clean_serving_state():
    feature_store.reset_backend_for_tests()
    model_cache.clear()
    yield
    feature_store.reset_backend_for_tests()
    model_cache.clear()


def _train(client, **overrides):
    payload = {"target": "cancellation", "model": "logistic_regression", "dataset_rows": 800}
    payload.update(overrides)
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_materialize_then_predict_by_id(client):
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 100})
    run_id = _train(client)

    response = client.post(
        f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001", "BK-000002"]}
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert [p["entity_id"] for p in body["predictions"]] == ["BK-000001", "BK-000002"]
    for prediction in body["predictions"]:
        assert prediction["features_found"] is True
        assert prediction["label"] in {"Cancelled", "Honoured"}
        assert prediction["imputed_features"] == 0


def test_diagnostics_split_lookup_from_inference(client):
    """A single latency number cannot tell you which layer is slow."""
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 50})
    run_id = _train(client)

    body = client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001"]}).json()
    diagnostics = body["diagnostics"]

    assert diagnostics["feature_lookup_ms"] >= 0
    assert diagnostics["inference_ms"] >= 0
    assert diagnostics["entities_requested"] == 1
    assert diagnostics["entities_found"] == 1
    assert diagnostics["backend"] in {"redis", "memory"}


def test_a_miss_still_returns_a_prediction_but_flags_it(client):
    """Degrade, don't fail — but never let a miss look like a real answer."""
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 10})
    run_id = _train(client)

    body = client.post(
        f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001", "BK-999999"]}
    ).json()

    hit, miss = body["predictions"]
    assert hit["features_found"] is True
    assert miss["features_found"] is False
    assert miss["prediction"] is not None          # still served, fully imputed
    assert body["diagnostics"]["entities_found"] == 1


def test_empty_store_is_a_total_miss_not_a_crash(client):
    run_id = _train(client)
    body = client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001"]}).json()

    assert body["predictions"][0]["features_found"] is False
    assert body["diagnostics"]["entities_found"] == 0


def test_schema_mismatch_returns_503(client, monkeypatch):
    """Feature layer unsafe → 503, not 500. The model is fine; the data is not."""
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 10})
    run_id = _train(client)
    monkeypatch.setattr(feature_store, "schema_hash", lambda names: "wrong-hash")

    response = client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001"]})
    assert response.status_code == 503
    assert "schema mismatch" in response.text


def test_predict_by_id_rejects_an_empty_batch(client):
    run_id = _train(client)
    assert client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": []}).status_code == 422


def test_deleting_a_run_evicts_its_cached_model(client):
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 10})
    run_id = _train(client)
    client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001"]})
    assert run_id in model_cache.stats()["cached_runs"]

    client.delete(f"/api/runs/{run_id}")
    assert run_id not in model_cache.stats()["cached_runs"]


def test_model_cache_hit_rate_climbs_with_reuse(client):
    client.post("/api/feature-store/materialize", json={"target": "cancellation", "limit": 10})
    run_id = _train(client)

    for _ in range(5):
        client.post(f"/api/runs/{run_id}/predict-by-id", json={"entity_ids": ["BK-000001"]})

    stats = client.get("/api/model-cache/stats").json()
    assert stats["loads"] == 1, "the model should be read from disk exactly once"
    assert stats["hits"] >= 4
