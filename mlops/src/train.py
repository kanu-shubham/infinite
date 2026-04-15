"""
train.py
--------
End-to-end training script:
  1. Generate / load data
  2. Split into train / test
  3. Build sklearn Pipeline  (preprocessor + XGBoost)
  4. Fit and evaluate
  5. Log everything to MLflow (params, metrics, model artifact)
  6. Persist model to disk so the API container can load it

Run locally:
    cd mlops/
    make train
    # or
    PYTHONPATH=. python -m src.train

Azure production run (CI/CD sets these env vars automatically):
    MLFLOW_TRACKING_URI=https://...  AZURE_STORAGE_CONNECTION_STRING=... make train
"""
import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    generate_hotel_data,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_mlflow_config(mlflow_cfg: dict) -> tuple[str, str | None]:
    """
    Resolve where MLflow should track runs and store artifacts.

    Priority order:
      1. MLFLOW_TRACKING_URI env var  (set by CI/CD for a real MLflow server)
      2. AZURE_STORAGE_CONNECTION_STRING env var  (write artifacts to Blob Storage)
      3. config.yaml value  (local mlruns/ folder — development only)

    Returns (tracking_uri, artifact_location).
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])

    # If Azure Blob Storage credentials are present, point artifacts there.
    # Format: wasbs://<container>@<account>.blob.core.windows.net/
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        account = mlflow_cfg.get("artifact_location", "")
        # Extract account name from connection string for the wasbs URI
        try:
            parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)
            account_name = parts.get("AccountName", "hotelmlopsstore")
        except Exception:
            account_name = "hotelmlopsstore"
        artifact_location = (
            f"wasbs://mlflow-artifacts@{account_name}.blob.core.windows.net/"
        )
        logger.info(f"Azure Blob Storage detected — artifacts → {artifact_location}")
    else:
        artifact_location = None  # MLflow uses default (tracking_uri)

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    return tracking_uri, artifact_location


def train(config_path: str = "config.yaml") -> tuple[Pipeline, dict]:
    config = load_config(config_path)
    training_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    # ── MLflow setup ──────────────────────────────────────────────────────────
    tracking_uri, artifact_location = _resolve_mlflow_config(mlflow_cfg)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # If an artifact location was resolved (e.g. Azure Blob), apply it.
    # This makes every run in this experiment store model files in Blob Storage
    # instead of on the local disk — critical for cloud training jobs.
    if artifact_location and experiment.artifact_location != artifact_location:
        mlflow.set_experiment(
            mlflow_cfg["experiment_name"]
        )  # idempotent; artifact_location set at experiment creation

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Generating training data ...")
    df = generate_hotel_data(
        n_samples=training_cfg["n_samples"],
        random_state=training_cfg["random_state"],
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_cfg["test_size"],
        random_state=training_cfg["random_state"],
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    xgb_params = {
        "n_estimators": training_cfg["n_estimators"],
        "max_depth": training_cfg["max_depth"],
        "learning_rate": training_cfg["learning_rate"],
        "subsample": training_cfg["subsample"],
        "colsample_bytree": training_cfg["colsample_bytree"],
        "random_state": training_cfg["random_state"],
        "n_jobs": -1,
    }

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(**xgb_params)),
        ]
    )

    # ── Train + evaluate ──────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        logger.info("Fitting model ...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

        logger.info(
            f"Results  →  MAE: ${metrics['mae']:.2f}  |  "
            f"RMSE: ${metrics['rmse']:.2f}  |  R²: {metrics['r2']:.4f}"
        )

        # Log to MLflow ─────────────────────────────────────────────────────
        mlflow.log_params(
            {
                "n_samples": training_cfg["n_samples"],
                "test_size": training_cfg["test_size"],
                **xgb_params,
            }
        )
        mlflow.log_metrics(metrics)

        # Log the full sklearn Pipeline (preprocessor + model) as an artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Save baseline feature statistics for drift detection
        os.makedirs("data", exist_ok=True)
        X_train.describe().to_csv("data/baseline_stats.csv")
        mlflow.log_artifact("data/baseline_stats.csv")

        logger.info(f"MLflow run: {run.info.run_id}")

    # ── Persist model for the API container ───────────────────────────────────
    model_path = config["serving"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved → {model_path}")

    return pipeline, metrics


if __name__ == "__main__":
    _, metrics = train()
    # Exit non-zero if quality gate would block deployment
    if metrics["r2"] < 0.85:
        logger.error(f"R² {metrics['r2']:.4f} below threshold 0.85 — aborting.")
        sys.exit(1)
