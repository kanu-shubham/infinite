"""
train.py
--------
Trains a hotel price prediction model and logs everything to MLflow.

WHAT IS MLFLOW?
  MLflow is an open-source platform for the complete ML lifecycle:
    - Tracking  : logs params, metrics, and artefacts for every run
    - Models    : packages models in a standard format (mlflow.models)
    - Registry  : versioned model store with Staging / Production stages
    - Projects  : reproducible ML code packaged for any platform

WHY TRACK EXPERIMENTS?
  Without tracking you'll ask yourself: "Which model did I deploy last week?
  What learning rate was it trained with?  Was it better than today's run?"
  MLflow answers all these questions automatically.

USAGE:
    python train.py                        # uses default hyperparams
    python train.py --n-estimators 200     # override via CLI

After running, open the MLflow UI:
    mlflow ui
    → http://localhost:5000
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from training.preprocess import load_data, get_splits, save_preprocessor

MLFLOW_EXPERIMENT = "hotel-price-prediction"
MODELS_DIR        = Path(__file__).parents[1] / "models" / "artefacts"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def train(
    n_estimators: int  = 300,
    max_depth:    int  = 6,
    learning_rate: float = 0.05,
    subsample:    float = 0.8,
    reg_alpha:    float = 0.1,   # L1 regularisation – reduces overfitting
    reg_lambda:   float = 1.0,   # L2 regularisation
):
    # ── Load and split data ───────────────────────────────────────────────────
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = get_splits(df)

    # ── Configure MLflow ─────────────────────────────────────────────────────
    # set_experiment creates the experiment if it doesn't exist yet.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        print(f"\nMLflow run ID: {run.info.run_id}")

        # ── Log hyperparameters ───────────────────────────────────────────────
        # Everything you log here will appear in the MLflow UI table,
        # making it trivial to compare hundreds of runs side-by-side.
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        mlflow.log_params(params)

        # ── Train ─────────────────────────────────────────────────────────────
        model = XGBRegressor(
            **params,
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=20,  # stop if val loss doesn't improve
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── Evaluate and log metrics ──────────────────────────────────────────
        # We log metrics on BOTH val and test sets.
        # val  → used to tune hyperparameters (early stopping etc.)
        # test → the final unbiased performance estimate for stakeholders
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            preds = model.predict(X)
            mlflow.log_metrics({
                f"{split_name}_rmse": rmse(y, preds),
                f"{split_name}_mae":  mae(y, preds),
                f"{split_name}_r2":   r2(y, preds),
            })
            print(f"  {split_name:4s} → RMSE={rmse(y, preds):.2f}  "
                  f"MAE={mae(y, preds):.2f}  R²={r2(y, preds):.4f}")

        # ── Log the actual best iteration MLflow found ─────────────────────────
        mlflow.log_metric("best_iteration", model.best_iteration)

        # ── Save artefacts ────────────────────────────────────────────────────
        # Artefacts are any files (models, plots, configs) stored alongside
        # the run so you can reproduce it exactly later.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        prep_path = save_preprocessor(preprocessor)
        mlflow.log_artifact(str(prep_path), artifact_path="preprocessor")

        # mlflow.xgboost.log_model stores the model in MLflow's model format.
        # This lets you load it later with mlflow.xgboost.load_model() or
        # even deploy it directly to a REST endpoint with `mlflow models serve`.
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="hotel-price-predictor",  # registers in Model Registry
        )

        print(f"\nRun logged. Open the UI:  mlflow ui")
        print(f"Model registered as:      hotel-price-predictor")

        return run.info.run_id, model, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hotel price prediction model")
    parser.add_argument("--n-estimators",  type=int,   default=300)
    parser.add_argument("--max-depth",     type=int,   default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample",     type=float, default=0.8)
    parser.add_argument("--reg-alpha",     type=float, default=0.1)
    parser.add_argument("--reg-lambda",    type=float, default=1.0)
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
    )
