"""
evaluate.py
-----------
Loads the latest production model from the MLflow Model Registry and
evaluates it on fresh data.

WHY a separate evaluate step?
  Training metrics can look great but fail to tell you:
    - Does the model still perform well on this week's data?
    - Is there feature drift? (distribution of inputs changed)
    - Should we promote a Staging model to Production?

  This script is run by the Airflow DAG after every training run and
  by the CI/CD pipeline on every PR to catch regressions before deploy.

MLFLOW MODEL REGISTRY STAGES:
  None       → newly registered, not yet validated
  Staging    → passed automated tests, ready for human review
  Production → live in the serving API
  Archived   → retired version kept for audit trail
"""

import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from training.preprocess import load_data, get_splits

EXPERIMENT_NAME   = "hotel-price-prediction"
MODEL_NAME        = "hotel-price-predictor"
RMSE_THRESHOLD    = 35.0   # USD – model is "good enough" if test RMSE < $35
R2_THRESHOLD      = 0.85   # model explains at least 85 % of price variance


def load_production_model():
    """Fetches the model currently in the Production stage."""
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        # Fall back to Staging if nothing is in Production yet
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging", "None"])
    if not versions:
        raise RuntimeError(f"No registered versions found for '{MODEL_NAME}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    print(f"Loading model: {MODEL_NAME} v{latest.version} (stage={latest.current_stage})")
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    return model, latest


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate():
    df = load_data()
    _, _, X_test, _, _, y_test, preprocessor = get_splits(df)

    model, version_info = load_production_model()
    preds = model.predict(X_test)

    test_rmse = rmse(y_test, preds)
    test_r2   = r2(y_test, preds)

    print(f"\nEvaluation Results")
    print(f"  Test RMSE : ${test_rmse:.2f}  (threshold < ${RMSE_THRESHOLD})")
    print(f"  Test R²   :  {test_r2:.4f}  (threshold > {R2_THRESHOLD})")

    passed = test_rmse < RMSE_THRESHOLD and test_r2 > R2_THRESHOLD

    if passed:
        print("\n  PASS – model meets quality gates.")
        # Automatically promote Staging → Production
        client = mlflow.tracking.MlflowClient()
        if version_info.current_stage != "Production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version_info.version,
                stage="Production",
            )
            print(f"  Promoted v{version_info.version} → Production")
    else:
        print("\n  FAIL – model does NOT meet quality gates. Blocking deploy.")
        sys.exit(1)   # non-zero exit code causes CI/CD pipeline to fail


if __name__ == "__main__":
    evaluate()
