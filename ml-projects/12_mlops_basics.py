"""
Project 12: MLOps Basics — Save, Load, Track, Reproduce
=========================================================
Production ML workflow: model persistence, reproducibility, and experiment tracking.

What you'll learn:
- Saving and loading models (joblib, pickle)
- Creating reproducible ML pipelines
- Experiment tracking with a lightweight logger (MLflow-style)
- Model versioning and comparison
- Building a reusable training pipeline
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Experiment Tracker (lightweight MLflow alternative) ──────────────────
class ExperimentTracker:
    """Simple experiment logger that saves runs to JSON files."""

    def __init__(self, experiment_name, base_dir="mlops_artifacts"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / experiment_name / "runs"
        self.models_dir = self.base_dir / experiment_name / "models"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_run = None

    def start_run(self, run_name):
        """Start a new experiment run."""
        self.current_run = {
            "run_name": run_name,
            "run_id": hashlib.md5(f"{run_name}{time.time()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        print(f"  [Run: {self.current_run['run_id']}] Started '{run_name}'")
        return self

    def log_param(self, key, value):
        """Log a parameter."""
        self.current_run["params"][key] = value

    def log_params(self, params_dict):
        """Log multiple parameters."""
        self.current_run["params"].update(params_dict)

    def log_metric(self, key, value):
        """Log a metric."""
        self.current_run["metrics"][key] = round(value, 6)

    def log_metrics(self, metrics_dict):
        """Log multiple metrics."""
        for k, v in metrics_dict.items():
            self.log_metric(k, v)

    def log_tag(self, key, value):
        """Log a tag."""
        self.current_run["tags"][key] = value

    def log_model(self, model, model_name):
        """Save the model as an artifact."""
        model_path = self.models_dir / f"{self.current_run['run_id']}_{model_name}.joblib"
        joblib.dump(model, model_path)
        self.current_run["artifacts"].append(str(model_path))
        print(f"  [Run: {self.current_run['run_id']}] Model saved to {model_path}")
        return model_path

    def end_run(self):
        """Save the run metadata."""
        run_path = self.runs_dir / f"{self.current_run['run_id']}.json"
        with open(run_path, "w") as f:
            json.dump(self.current_run, f, indent=2)
        print(f"  [Run: {self.current_run['run_id']}] Run logged to {run_path}\n")
        return self.current_run

    def get_all_runs(self):
        """Load all runs for this experiment."""
        runs = []
        for run_file in sorted(self.runs_dir.glob("*.json")):
            with open(run_file) as f:
                runs.append(json.load(f))
        return runs

    def get_best_run(self, metric, minimize=True):
        """Find the best run by a given metric."""
        runs = self.get_all_runs()
        if not runs:
            return None
        key_fn = lambda r: r["metrics"].get(metric, float("inf") if minimize else float("-inf"))
        return min(runs, key=key_fn) if minimize else max(runs, key=key_fn)


# ── Reproducible Pipeline Builder ────────────────────────────────────────
class MLPipeline:
    """Reusable ML pipeline with built-in reproducibility."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.config = {
            "random_state": random_state,
            "created_at": datetime.now().isoformat(),
        }

    def load_data(self, test_size=0.2):
        """Load and split data reproducibly."""
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = housing.target

        self.feature_names = housing.feature_names
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.config["dataset"] = "california_housing"
        self.config["test_size"] = test_size
        self.config["n_train"] = len(self.X_train)
        self.config["n_test"] = len(self.X_test)
        self.config["n_features"] = X.shape[1]

        # Data fingerprint for reproducibility verification
        data_hash = hashlib.md5(X.values.tobytes()).hexdigest()[:12]
        self.config["data_hash"] = data_hash

        return self

    def build_pipeline(self, model, model_name="model"):
        """Build a sklearn Pipeline with scaling + model."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        self.model_name = model_name
        return self

    def train(self):
        """Train the pipeline."""
        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        """Evaluate and return metrics."""
        y_pred = self.pipeline.predict(self.X_test)
        self.metrics = {
            "mae": mean_absolute_error(self.y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
            "r2": r2_score(self.y_test, y_pred),
        }
        self.predictions = y_pred
        return self.metrics

    def cross_validate(self, cv=5):
        """Run cross-validation."""
        scores = cross_val_score(
            self.pipeline, self.X_train, self.y_train,
            cv=cv, scoring="neg_mean_absolute_error"
        )
        self.cv_scores = -scores
        self.metrics["cv_mae_mean"] = self.cv_scores.mean()
        self.metrics["cv_mae_std"] = self.cv_scores.std()
        return self.cv_scores


def main():
    # ── 1. Setup ─────────────────────────────────────────────────────────
    print("=== MLOps Basics: Save, Load, Track, Reproduce ===\n")

    tracker = ExperimentTracker("house_price_experiment")

    # ── 2. Model saving and loading ──────────────────────────────────────
    print("── Part 1: Model Persistence ──\n")

    pipeline = MLPipeline(random_state=42)
    pipeline.load_data(test_size=0.2)
    pipeline.build_pipeline(
        GradientBoostingRegressor(n_estimators=200, random_state=42),
        model_name="gradient_boosting",
    )
    pipeline.train()
    metrics = pipeline.evaluate()

    print(f"Trained model — MAE: ${metrics['mae'] * 100_000:,.0f}")

    # Save with joblib
    save_path = "mlops_artifacts/demo_model.joblib"
    os.makedirs("mlops_artifacts", exist_ok=True)
    joblib.dump(pipeline.pipeline, save_path)
    print(f"Saved model to: {save_path}")

    # Save the config for reproducibility
    config_path = "mlops_artifacts/demo_config.json"
    with open(config_path, "w") as f:
        json.dump(pipeline.config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Load and verify
    loaded_pipeline = joblib.load(save_path)
    loaded_pred = loaded_pipeline.predict(pipeline.X_test)
    original_pred = pipeline.pipeline.predict(pipeline.X_test)

    predictions_match = np.allclose(loaded_pred, original_pred)
    print(f"Loaded model predictions match: {predictions_match}")
    print(f"Model file size: {os.path.getsize(save_path) / 1024:.1f} KB\n")

    # ── 3. Experiment tracking ───────────────────────────────────────────
    print("── Part 2: Experiment Tracking ──\n")

    models_to_test = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {"type": "linear"},
        },
        "Ridge (alpha=1.0)": {
            "model": Ridge(alpha=1.0, random_state=42),
            "params": {"type": "ridge", "alpha": 1.0},
        },
        "Lasso (alpha=0.01)": {
            "model": Lasso(alpha=0.01, random_state=42),
            "params": {"type": "lasso", "alpha": 0.01},
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            "params": {"type": "random_forest", "n_estimators": 200, "max_depth": 15},
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
            "params": {"type": "gradient_boosting", "n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        },
        "GB (tuned)": {
            "model": GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
            "params": {"type": "gradient_boosting", "n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        },
    }

    for name, config in models_to_test.items():
        tracker.start_run(name)

        # Build and train
        p = MLPipeline(random_state=42)
        p.load_data(test_size=0.2)
        p.build_pipeline(config["model"], model_name=name)
        p.train()

        # Evaluate
        metrics = p.evaluate()
        cv_scores = p.cross_validate(cv=5)

        # Log everything
        tracker.log_params(config["params"])
        tracker.log_param("random_state", 42)
        tracker.log_metrics(metrics)
        tracker.log_tag("stage", "experiment")
        tracker.log_model(p.pipeline, name.replace(" ", "_").lower())
        tracker.end_run()

    # ── 4. Compare experiments ───────────────────────────────────────────
    print("── Part 3: Experiment Comparison ──\n")

    all_runs = tracker.get_all_runs()

    print(f"{'Run Name':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'CV MAE':>10}")
    print("-" * 68)
    for run in all_runs:
        m = run["metrics"]
        print(f"{run['run_name']:<25} {m['mae']:>9.4f} {m['rmse']:>9.4f} {m['r2']:>7.4f} {m.get('cv_mae_mean', 0):>9.4f}")

    # Find best
    best_run = tracker.get_best_run("mae", minimize=True)
    print(f"\nBest model: {best_run['run_name']} (MAE: {best_run['metrics']['mae']:.4f})")
    print(f"Parameters: {best_run['params']}")

    # ── 5. Load the best model ───────────────────────────────────────────
    print(f"\n── Part 4: Loading Best Model ──\n")

    best_model_path = best_run["artifacts"][0]
    production_model = joblib.load(best_model_path)

    # Verify it works
    p_verify = MLPipeline(random_state=42)
    p_verify.load_data(test_size=0.2)
    verify_pred = production_model.predict(p_verify.X_test)
    verify_mae = mean_absolute_error(p_verify.y_test, verify_pred)
    print(f"Loaded best model from: {best_model_path}")
    print(f"Verification MAE: {verify_mae:.4f} (matches logged: {abs(verify_mae - best_run['metrics']['mae']) < 1e-6})")

    # ── 6. Reproducibility check ─────────────────────────────────────────
    print(f"\n── Part 5: Reproducibility Verification ──\n")

    # Run the same experiment twice
    results_1, results_2 = [], []
    for seed in [42, 42]:
        p = MLPipeline(random_state=seed)
        p.load_data(test_size=0.2)
        p.build_pipeline(GradientBoostingRegressor(n_estimators=200, random_state=seed))
        p.train()
        m = p.evaluate()
        if len(results_1) == 0:
            results_1 = p.pipeline.predict(p.X_test)
        else:
            results_2 = p.pipeline.predict(p.X_test)

    reproducible = np.allclose(results_1, results_2)
    print(f"Same seed → same results: {reproducible}")

    # Different seeds
    p_diff = MLPipeline(random_state=123)
    p_diff.load_data(test_size=0.2)
    p_diff.build_pipeline(GradientBoostingRegressor(n_estimators=200, random_state=123))
    p_diff.train()
    m_diff = p_diff.evaluate()

    print(f"seed=42  MAE: {verify_mae:.4f}")
    print(f"seed=123 MAE: {m_diff['mae']:.4f}")
    print(f"Different seeds → different results (expected)")

    # ── 7. Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Experiment comparison
    run_names = [r["run_name"] for r in all_runs]
    maes = [r["metrics"]["mae"] for r in all_runs]
    r2s = [r["metrics"]["r2"] for r in all_runs]

    colors = ["#4CAF50" if m == min(maes) else "#2196F3" for m in maes]
    axes[0, 0].barh(run_names, maes, color=colors)
    axes[0, 0].set_xlabel("MAE (lower = better)")
    axes[0, 0].set_title("Experiment Comparison: MAE")

    colors = ["#4CAF50" if r == max(r2s) else "#2196F3" for r in r2s]
    axes[0, 1].barh(run_names, r2s, color=colors)
    axes[0, 1].set_xlabel("R² (higher = better)")
    axes[0, 1].set_title("Experiment Comparison: R²")

    # Best model predictions
    axes[1, 0].scatter(p_verify.y_test, verify_pred, alpha=0.3, s=10)
    lims = [p_verify.y_test.min(), p_verify.y_test.max()]
    axes[1, 0].plot(lims, lims, "r--", linewidth=2)
    axes[1, 0].set_xlabel("Actual Price ($100k)")
    axes[1, 0].set_ylabel("Predicted Price ($100k)")
    axes[1, 0].set_title(f"Best Model: {best_run['run_name']}")

    # Run timeline
    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in all_runs]
    elapsed = [(t - timestamps[0]).total_seconds() for t in timestamps]
    axes[1, 1].plot(range(len(all_runs)), maes, "bo-", markersize=8)
    for i, (name, mae) in enumerate(zip(run_names, maes)):
        axes[1, 1].annotate(name, (i, mae), fontsize=7, rotation=20, ha="left")
    axes[1, 1].set_xlabel("Run #")
    axes[1, 1].set_ylabel("MAE")
    axes[1, 1].set_title("Experiment Progress")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("MLOps Dashboard", fontsize=14)
    plt.tight_layout()
    plt.savefig("12_mlops_dashboard.png", dpi=100)
    print(f"\nSaved dashboard to 12_mlops_dashboard.png")

    # ── 8. Summary ───────────────────────────────────────────────────────
    print(f"\n=== Artifacts Created ===")
    artifacts_dir = Path("mlops_artifacts")
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            unit = "KB" if size > 1024 else "B"
            size_display = size / 1024 if size > 1024 else size
            print(f"  {path} ({size_display:.1f} {unit})")

    print(f"\n=== Key MLOps Lessons ===")
    print("1. Always save models with joblib/pickle for deployment")
    print("2. Track experiments — parameters, metrics, and artifacts")
    print("3. Set random seeds everywhere for reproducibility")
    print("4. Version your data (data hash) alongside your model")
    print("5. Use pipelines to bundle preprocessing + model together")
    print("6. Compare multiple runs to find the best configuration")


if __name__ == "__main__":
    main()
