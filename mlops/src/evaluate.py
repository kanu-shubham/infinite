"""
evaluate.py
-----------
Model quality gate – run after training to decide whether the model is
good enough to deploy.

In a real system this would also:
  - Compare the new model against the currently-deployed "champion" model
  - Run statistical tests (e.g. Kolmogorov-Smirnov) for data drift
  - Log results to a monitoring dashboard (Grafana, Weights & Biases, etc.)

CI pipeline calls this script and fails the build if the gate doesn't pass.

Run locally:
    cd mlops/
    make evaluate
"""
import logging
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, generate_hotel_data, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str = "models/model.pkl",
    config_path: str = "config.yaml",
) -> dict:
    config = load_config(config_path)
    threshold = config["monitoring"]["min_r2_threshold"]

    logger.info(f"Loading model from {model_path} ...")
    pipeline = joblib.load(model_path)

    # Use a separate random seed so this is genuinely held-out data
    df = generate_hotel_data(n_samples=3_000, random_state=999)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    y_pred = pipeline.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }

    logger.info("── Evaluation Results ────────────────────────────────")
    logger.info(f"  MAE  : ${metrics['mae']:.2f}")
    logger.info(f"  RMSE : ${metrics['rmse']:.2f}")
    logger.info(f"  R²   : {metrics['r2']:.4f}  (threshold: {threshold})")
    logger.info("──────────────────────────────────────────────────────")

    if metrics["r2"] < threshold:
        logger.error(
            f"QUALITY GATE FAILED — R² {metrics['r2']:.4f} < {threshold}. "
            "Model will NOT be promoted to production."
        )
        sys.exit(1)

    logger.info("QUALITY GATE PASSED — model is ready for deployment.")
    return metrics


if __name__ == "__main__":
    evaluate_model()
