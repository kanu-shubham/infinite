"""End-to-end API tests.

TestClient drains BackgroundTasks before the response context closes, so a
run has already finished by the time the POST returns.
"""


def _train(client, **overrides):
    payload = {"target": "cancellation", "model": "logistic_regression", "dataset_rows": 800}
    payload.update(overrides)
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_health_reports_ok(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_catalog_lists_models_matching_each_target_task(client):
    catalog = client.get("/api/catalog").json()
    for target in catalog["targets"]:
        assert target["models"], f"no models for {target['key']}"
        assert all(model["task"] == target["task"] for model in target["models"])
    assert len(catalog["stages"]) == 8


def test_dataset_endpoint_describes_features_and_preview(client):
    body = client.get("/api/dataset", params={"target": "cancellation"}).json()
    assert body["profile"]["rows"] > 0
    assert len(body["preview"]) == 5
    assert {spec["kind"] for spec in body["features"]} == {"numeric", "categorical"}


def test_dataset_rejects_an_unknown_target(client):
    assert client.get("/api/dataset", params={"target": "nope"}).status_code == 422


def test_training_run_succeeds_and_reports_every_stage(client):
    run_id = _train(client)
    run = client.get(f"/api/runs/{run_id}").json()

    assert run["status"] == "succeeded", run.get("error")
    assert run["progress"] == 1.0
    assert all(stage["status"] in {"succeeded", "skipped"} for stage in run["stages"])
    assert run["metrics"]["scores"]["accuracy"] > 0.5
    assert run["headline_metric"]["value"] is not None
    assert run["feature_importances"]
    assert run["dataset"]["train_rows"] + run["dataset"]["test_rows"] == run["dataset"]["rows"]
    assert run["has_model"] is True


def test_cross_validation_runs_when_folds_are_requested(client):
    run_id = _train(client, cv_folds=3)
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["cross_validation"]["folds"] == 3
    assert len(run["cross_validation"]["scores"]) == 3


def test_cross_validation_stage_is_skipped_by_default(client):
    run_id = _train(client)
    run = client.get(f"/api/runs/{run_id}").json()
    stage = next(s for s in run["stages"] if s["key"] == "cross_validate")
    assert stage["status"] == "skipped"
    assert run["cross_validation"] is None


def test_regression_run_produces_regression_metrics(client):
    run_id = _train(client, target="price", model="ridge")
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "succeeded", run.get("error")
    assert set(run["metrics"]["scores"]) == {"r2", "mae", "rmse", "mape"}
    assert run["metrics"]["scatter"]


def test_model_task_must_match_the_target_task(client):
    response = client.post("/api/runs", json={"target": "price", "model": "logistic_regression"})
    assert response.status_code == 422
    assert "regression" in response.text


def test_unknown_model_is_rejected(client):
    response = client.post("/api/runs", json={"target": "cancellation", "model": "xgboost"})
    assert response.status_code == 422


def test_single_fold_cross_validation_is_rejected(client):
    response = client.post("/api/runs", json={"target": "cancellation", "cv_folds": 1})
    assert response.status_code == 422


def test_runs_are_listed_newest_first_and_can_be_filtered(client):
    first = _train(client)
    second = _train(client, name="second run")

    runs = client.get("/api/runs").json()["runs"]
    assert [run["run_id"] for run in runs][:2] == [second, first]

    succeeded = client.get("/api/runs", params={"status": "succeeded"}).json()["runs"]
    assert len(succeeded) == 2

    assert client.get("/api/runs", params={"status": "failed"}).json()["runs"] == []


def test_run_summaries_omit_the_heavy_detail_payload(client):
    _train(client)
    summary = client.get("/api/runs").json()["runs"][0]
    assert "logs" not in summary and "metrics" not in summary
    assert summary["headline_metric"]["value"] is not None


def test_prediction_uses_the_registered_model(client):
    run_id = _train(client)
    sample = client.get("/api/dataset/sample", params={"count": 2}).json()["rows"]

    response = client.post(f"/api/runs/{run_id}/predict", json={"rows": sample})
    assert response.status_code == 200, response.text

    body = response.json()
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert prediction["label"] in {"Cancelled", "Honoured"}
        assert 0.0 <= prediction["probability"] <= 1.0


def test_prediction_fills_in_omitted_features(client):
    run_id = _train(client)
    response = client.post(f"/api/runs/{run_id}/predict", json={"rows": [{"lead_time": 250}]})
    assert response.status_code == 200, response.text
    assert response.json()["predictions"][0]["prediction"] in (0.0, 1.0)


def test_regression_predictions_have_no_class_label(client):
    run_id = _train(client, target="price", model="ridge")
    response = client.post(f"/api/runs/{run_id}/predict", json={"rows": [{"adults": 2}]})
    prediction = response.json()["predictions"][0]
    assert prediction["label"] is None
    assert prediction["prediction"] > 0


def test_prediction_requires_at_least_one_row(client):
    run_id = _train(client)
    assert client.post(f"/api/runs/{run_id}/predict", json={"rows": []}).status_code == 422


def test_prediction_on_an_unknown_run_is_a_404(client):
    assert client.post("/api/runs/run-missing/predict", json={"rows": [{}]}).status_code == 404


def test_deleting_a_run_removes_it_and_its_model(client):
    run_id = _train(client)
    assert client.delete(f"/api/runs/{run_id}").json()["deleted"] is True
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.delete(f"/api/runs/{run_id}").status_code == 404


def test_run_survives_a_registry_reload(client):
    from app.ml import registry

    run_id = _train(client)
    registry.reset_for_tests()
    registry.load_from_disk()

    reloaded = registry.get(run_id)
    assert reloaded is not None and reloaded["status"] == "succeeded"
