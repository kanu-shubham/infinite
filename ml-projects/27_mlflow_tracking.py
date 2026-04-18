"""
Project 27: Real MLflow Experiment Tracking
=============================================
MLflow is the industry-standard tool for tracking ML experiments.
It replaces the toy JSON tracker from Project 12 with a full UI,
artifact storage, model registry, and deployment capabilities.

What you'll learn:
- MLflow tracking: log params, metrics, artifacts to a local server
- MLflow UI: browse experiments in a web browser
- Model registry: version and stage models (Staging → Production)
- MLflow Projects: reproducible experiment packaging
- Comparing runs programmatically
- Auto-logging: one line to capture everything
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import infer_signature

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Setup ─────────────────────────────────────────────────────────────────

def setup_mlflow(experiment_name="house_price_regression"):
    """
    Set up MLflow tracking.

    By default, MLflow logs to ./mlruns/ directory.
    To launch the UI:
        mlflow ui --port 5000
    Then open: http://localhost:5000
    """
    mlflow.set_tracking_uri("mlruns")          # local file-based tracking
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    print(f"View UI with: mlflow ui --port 5000\n")


def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    return X_train_s, X_test_s, y_train, y_test, scaler, housing.feature_names


# ── Part 1: Manual logging ────────────────────────────────────────────────

def run_experiment_manual(X_train, X_test, y_train, y_test, feature_names):
    """
    Manual MLflow logging — full control over what gets tracked.
    """
    print("── Part 1: Manual MLflow Logging ──\n")

    model_configs = [
        {
            "name": "Ridge Regression",
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0, "model_type": "ridge"},
            "tags": {"team": "baseline"},
        },
        {
            "name": "Random Forest",
            "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10, "model_type": "random_forest"},
            "tags": {"team": "ml_team"},
        },
        {
            "name": "Gradient Boosting",
            "model": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                               learning_rate=0.1, random_state=42),
            "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                       "model_type": "gradient_boosting"},
            "tags": {"team": "ml_team"},
        },
        {
            "name": "GB Tuned",
            "model": GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42),
            "params": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
                       "subsample": 0.8, "model_type": "gradient_boosting_tuned"},
            "tags": {"team": "ml_team", "tuned": "true"},
        },
    ]

    run_ids = []

    for cfg in model_configs:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            run_id = run.info.run_id
            run_ids.append(run_id)

            # ── Log parameters ─────────────────────────────────────────
            mlflow.log_params(cfg["params"])
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)

            # ── Log tags ───────────────────────────────────────────────
            mlflow.set_tags(cfg["tags"])
            mlflow.set_tag("dataset", "california_housing")

            # ── Train ──────────────────────────────────────────────────
            model = cfg["model"]
            model.fit(X_train, y_train)

            # ── Log metrics ────────────────────────────────────────────
            y_pred = model.predict(X_test)
            mae    = mean_absolute_error(y_test, y_pred)
            rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
            r2     = r2_score(y_test, y_pred)

            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2",   r2)

            # ── Log artifact: feature importance plot ──────────────────
            if hasattr(model, "feature_importances_"):
                fig, ax = plt.subplots(figsize=(8, 5))
                importances = model.feature_importances_
                idx = importances.argsort()
                ax.barh([feature_names[i] for i in idx], importances[idx])
                ax.set_title(f"Feature Importance — {cfg['name']}")
                plt.tight_layout()
                fig_path = f"/tmp/feat_importance_{cfg['name'].replace(' ', '_')}.png"
                fig.savefig(fig_path)
                plt.close(fig)
                mlflow.log_artifact(fig_path, artifact_path="plots")

            # ── Log the model itself ───────────────────────────────────
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                signature=signature,
                registered_model_name=cfg["name"].replace(" ", "_"),
            )

            print(f"  {cfg['name']:<22} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
                  f"R²: {r2:.4f} | run_id: {run_id[:8]}...")

    print()
    return run_ids


# ── Part 2: Auto-logging ──────────────────────────────────────────────────

def run_experiment_autolog(X_train, X_test, y_train, y_test):
    """
    mlflow.sklearn.autolog() captures everything automatically:
    parameters, metrics, model, feature importance — one line.
    """
    print("── Part 2: Auto-Logging (one line captures everything) ──\n")

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=True,
    )

    models = {
        "AutoLog_RF":  RandomForestRegressor(n_estimators=50, random_state=42),
        "AutoLog_GB":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # autolog already captured train metrics; add test metrics manually
            mlflow.log_metrics({
                "test_mae":  mean_absolute_error(y_test, y_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_r2":   r2_score(y_test, y_pred),
            })
            print(f"  {name} logged automatically")

    mlflow.sklearn.autolog(disable=True)
    print()


# ── Part 3: Iterative metric logging ─────────────────────────────────────

def run_iterative_logging(X_train, X_test, y_train, y_test):
    """
    Log metrics at each step (epoch/iteration) — shows learning curves in UI.
    """
    print("── Part 3: Iterative Metric Logging (training curves) ──\n")

    with mlflow.start_run(run_name="GB_incremental"):
        mlflow.log_param("model_type", "gradient_boosting_incremental")
        mlflow.log_param("max_n_estimators", 200)

        model = GradientBoostingRegressor(
            n_estimators=1, warm_start=True, random_state=42, max_depth=5
        )

        step_sizes = list(range(10, 210, 10))
        for step, n in enumerate(step_sizes):
            model.n_estimators = n
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mlflow.log_metric("test_mae",  mean_absolute_error(y_test, y_pred), step=step)
            mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test, y_pred)), step=step)
            mlflow.log_metric("test_r2",   r2_score(y_test, y_pred), step=step)
            mlflow.log_metric("n_estimators", n, step=step)

        final_mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"  Final MAE with {max(step_sizes)} estimators: {final_mae:.4f}")
        mlflow.sklearn.log_model(model, "model")
    print()


# ── Part 4: Compare runs programmatically ────────────────────────────────

def compare_runs():
    """Load all runs and compare them without opening the UI."""
    print("── Part 4: Programmatic Run Comparison ──\n")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("house_price_regression")

    if experiment is None:
        print("No experiments found yet.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
    )

    print(f"Total runs: {len(runs)}\n")
    print(f"{'Run Name':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 58)

    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        mae  = run.data.metrics.get("mae", run.data.metrics.get("test_mae", float("nan")))
        rmse = run.data.metrics.get("rmse", run.data.metrics.get("test_rmse", float("nan")))
        r2   = run.data.metrics.get("r2", run.data.metrics.get("test_r2", float("nan")))
        print(f"{name:<28} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}")

    if runs:
        best = runs[0]
        best_name = best.data.tags.get("mlflow.runName", "best")
        best_mae  = best.data.metrics.get("mae", best.data.metrics.get("test_mae"))
        print(f"\nBest run: '{best_name}' (MAE: {best_mae:.4f})")
        print(f"Run ID: {best.info.run_id}")


# ── Part 5: Model registry ────────────────────────────────────────────────

def demo_model_registry():
    """
    MLflow Model Registry: version control for ML models.
    Stages: None → Staging → Production → Archived
    """
    print("\n── Part 5: Model Registry Concepts ──\n")
    print("""
  MLflow Model Registry workflow:

  1. Log model during training:
       mlflow.sklearn.log_model(model, "model",
                                registered_model_name="HousePriceModel")

  2. View versions in UI: Models tab → HousePriceModel

  3. Transition to staging:
       client.transition_model_version_stage(
           name="HousePriceModel", version=1, stage="Staging"
       )

  4. Load for inference:
       model = mlflow.pyfunc.load_model(
           "models:/HousePriceModel/Production"
       )
       predictions = model.predict(X_test)

  5. Promote to production after validation:
       client.transition_model_version_stage(
           name="HousePriceModel", version=2, stage="Production"
       )

  This gives you:
    ✓ Model versioning (v1, v2, v3...)
    ✓ Rollback capability (revert to v1 if v2 is bad)
    ✓ Stage-based deployment (staging vs production)
    ✓ Model lineage (which code + data produced this model)
    """)


def main():
    print("=== Real MLflow Experiment Tracking ===\n")

    print("── What MLflow Tracks ──\n")
    print("  Parameters:  hyperparameters (n_estimators=100, lr=0.01)")
    print("  Metrics:     MAE, RMSE, R², accuracy — by step or epoch")
    print("  Artifacts:   model files, plots, confusion matrices, data")
    print("  Tags:        team, dataset, notes, commit hash")
    print("  Models:      versioned model registry with staging/production\n")

    setup_mlflow()

    X_train, X_test, y_train, y_test, scaler, feature_names = load_data()
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples\n")

    run_experiment_manual(X_train, X_test, y_train, y_test, feature_names)
    run_experiment_autolog(X_train, X_test, y_train, y_test)
    run_iterative_logging(X_train, X_test, y_train, y_test)
    compare_runs()
    demo_model_registry()

    print("\n── To View the MLflow UI ──\n")
    print("  pip install mlflow")
    print("  mlflow ui --port 5000")
    print("  Open: http://localhost:5000")
    print("\n  You'll see:")
    print("    - All experiments and runs in a table")
    print("    - Compare any two runs side-by-side")
    print("    - Learning curves (metric vs step)")
    print("    - Artifact browser (plots, models)")
    print("    - Model registry with version history")

    print("\n── MLflow vs Weights & Biases ──\n")
    print("  MLflow:    open-source, self-hosted, great for on-premise")
    print("  W&B:       cloud-based, richer UI, free for individuals")
    print("  Both:      log params, metrics, artifacts, models")
    print("  W&B extra: system metrics (GPU/CPU), rich media logging")
    print("  W&B usage: wandb.init() → wandb.log({...}) → wandb.finish()")

    print("\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ mlflow.start_run() — context manager for runs")
    print("  ✓ log_param/log_params — hyperparameter tracking")
    print("  ✓ log_metric — scalar metric (supports step for curves)")
    print("  ✓ log_artifact — save files (plots, configs, data)")
    print("  ✓ log_model + infer_signature — model + I/O schema")
    print("  ✓ autolog — one-liner to capture everything")
    print("  ✓ MlflowClient — programmatic run comparison")
    print("  ✓ Model Registry — version + stage ML models")


if __name__ == "__main__":
    main()
