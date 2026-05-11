"""
Train a hotel click-probability model and persist the artefacts.

The model predicts P(user clicks hotel | features) — a standard ranking
signal used in recommendation engines.  We use a Gradient Boosted Tree
because it is representative of what you'd find in production:
  - non-linear, captures price × rating interactions
  - inference is CPU-bound and releases the GIL via numpy
  - latency per prediction is ~1-5 ms — measurable but fast enough to
    demonstrate batching / caching effects at moderate concurrency

Run once before starting the server:
    python train.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "model.pkl"
N_SAMPLES   = 60_000
RANDOM_SEED = 42

FEATURE_NAMES = ["price", "rating", "review_count", "amenity_count", "location_score"]


def generate_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic hotel catalogue + noisy click labels.

    Ground-truth click probability follows a logistic model that encodes
    reasonable domain logic: cheap, highly-rated, well-reviewed hotels in
    central locations get clicked more.
    """
    rng = np.random.default_rng(seed)

    price          = rng.uniform(50, 800, n)          # USD / night
    rating         = rng.uniform(1.0, 5.0, n)
    review_count   = rng.integers(1, 5_000, n).astype(float)
    amenity_count  = rng.integers(0, 20, n).astype(float)
    location_score = rng.uniform(0.0, 1.0, n)         # 0 = remote, 1 = city-centre

    X = np.column_stack([price, rating, review_count, amenity_count, location_score])

    # True log-odds that a user clicks on this hotel impression
    log_odds = (
        -0.004 * price           # higher price → fewer clicks
        + 1.2  * rating          # rating is strong signal
        + 0.0003 * review_count  # social proof
        + 0.08 * amenity_count
        + 0.6  * location_score
        - 5.5                    # intercept: ~15% base click rate
    )
    p_click = 1 / (1 + np.exp(-log_odds))
    y = rng.binomial(1, p_click, n)

    return X, y


def main() -> None:
    print(f"Generating {N_SAMPLES:,} samples …")
    X, y = generate_dataset(N_SAMPLES, RANDOM_SEED)
    print(f"Click rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("Training GradientBoostingClassifier (100 trees, depth 4) …")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    model.fit(X_train_s, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    print(f"Test AUC: {auc:.4f}")

    artefacts = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "version":       "v1.0.0",
        "train_auc":     auc,
    }
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artefacts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
