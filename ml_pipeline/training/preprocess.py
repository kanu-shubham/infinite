"""
preprocess.py
-------------
Turns raw CSV data into a clean numpy array ready for model training.

This step is intentionally kept separate from training so that:
  - The same preprocessing can be used at serving time (no skew).
  - Preprocessing can be tested independently.
  - MLflow can log the preprocessor as an artefact alongside the model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


DATA_PATH  = Path(__file__).parents[1] / "data" / "hotels.csv"
MODELS_DIR = Path(__file__).parents[1] / "models" / "artefacts"

CATEGORICAL_FEATURES = ["city", "category"]
NUMERIC_FEATURES     = [
    "star_rating", "review_score", "num_reviews",
    "distance_km", "amenities", "rooms_available",
]
TARGET = "price_usd"

TEST_SIZE   = 0.20
VALID_SIZE  = 0.10   # fraction of total, taken from the remaining train split
RANDOM_SEED = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop columns that must NOT be used as features (target leakage prevention).
    df = df.drop(columns=["hotel_id", "event_timestamp"], errors="ignore")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a scikit-learn ColumnTransformer that:
      - One-hot encodes categorical columns  (city, category)
      - Standard-scales numeric columns      (mean=0, std=1)

    WHY standard-scale numerics?
      Tree models (XGBoost, RandomForest) don't strictly need scaling.
      Linear models and neural nets do.  Scaling here makes this pipeline
      work for any model type without changes.
    """
    numeric_pipe     = Pipeline([("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False))])

    return ColumnTransformer([
        ("num", numeric_pipe,     NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def get_splits(df: pd.DataFrame):
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor).

    The preprocessor is FIT only on X_train to avoid data leakage.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    X_train_raw, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train_raw, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train,
        test_size=VALID_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)   # fit only on train!
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = None):
    path = path or (MODELS_DIR / "preprocessor.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved → {path}")
    return path


if __name__ == "__main__":
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = get_splits(df)
    save_preprocessor(preprocessor)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
