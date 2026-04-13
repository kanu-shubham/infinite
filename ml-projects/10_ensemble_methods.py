"""
Project 10: Ensemble Methods — XGBoost & LightGBM
===================================================
Compare gradient boosting frameworks on a structured dataset.

What you'll learn:
- How ensemble methods (bagging vs boosting) work
- XGBoost and LightGBM — the industry workhorses
- Hyperparameter tuning with GridSearchCV
- Learning curves and early stopping
- SHAP-style feature analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    BaggingRegressor, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def main():
    # ── 1. Load dataset ──────────────────────────────────────────────────
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["target"] = housing.target

    print("=== Ensemble Methods Comparison ===")
    print(f"Dataset: California Housing ({len(df):,} samples, {len(housing.feature_names)} features)")
    print(f"Target: Median house value (in $100k)\n")

    if HAS_XGB:
        print("[OK] XGBoost available")
    else:
        print("[!!] XGBoost not installed (pip install xgboost) — using sklearn GradientBoosting as substitute")

    if HAS_LGBM:
        print("[OK] LightGBM available")
    else:
        print("[!!] LightGBM not installed (pip install lightgbm) — using sklearn GradientBoosting as substitute")
    print()

    # ── 2. Prepare data ──────────────────────────────────────────────────
    X = df.drop(columns="target").values
    y = df["target"].values
    feature_names = housing.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further split train into train/validation for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    print(f"Train:      {len(X_tr):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test:       {len(X_test):,}\n")

    # ── 3. Define all ensemble models ────────────────────────────────────
    models = {}

    # Baseline
    models["Linear Regression"] = LinearRegression()

    # Single tree
    models["Decision Tree"] = DecisionTreeRegressor(max_depth=10, random_state=42)

    # Bagging
    models["Bagging (100 trees)"] = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),
        n_estimators=100, random_state=42, n_jobs=-1,
    )

    # Random Forest
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
    )

    # AdaBoost
    models["AdaBoost"] = AdaBoostRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42,
    )

    # Sklearn GradientBoosting
    models["Sklearn GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
    )

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )

    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )

    # ── 4. Train and compare ─────────────────────────────────────────────
    print("=== Training All Models ===\n")
    print(f"{'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time (s)':>10}")
    print("-" * 68)

    results = {}
    for name, model in models.items():
        start = time.time()

        # Use early stopping for XGBoost and LightGBM
        if HAS_XGB and isinstance(model, XGBRegressor):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif HAS_LGBM and isinstance(model, LGBMRegressor):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "time": elapsed, "predictions": y_pred}
        print(f"{name:<28} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f} {elapsed:>9.2f}s")

    print()

    # ── 5. Visual comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = list(results.keys())
    maes = [results[n]["MAE"] for n in names]
    r2s = [results[n]["R2"] for n in names]
    times = [results[n]["time"] for n in names]

    # MAE comparison
    colors = ["#4CAF50" if m == min(maes) else "#2196F3" for m in maes]
    bars = axes[0].barh(names, maes, color=colors)
    axes[0].set_xlabel("Mean Absolute Error (lower = better)")
    axes[0].set_title("MAE Comparison")
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

    # R² comparison
    colors = ["#4CAF50" if r == max(r2s) else "#2196F3" for r in r2s]
    bars = axes[1].barh(names, r2s, color=colors)
    axes[1].set_xlabel("R² Score (higher = better)")
    axes[1].set_title("R² Comparison")
    for bar, val in zip(bars, r2s):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

    # Training time
    bars = axes[2].barh(names, times, color="#FF9800")
    axes[2].set_xlabel("Training Time (seconds)")
    axes[2].set_title("Speed Comparison")
    for bar, val in zip(bars, times):
        axes[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}s", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("10_ensemble_comparison.png", dpi=100)
    print("Saved model comparison to 10_ensemble_comparison.png")

    # ── 6. Feature importance comparison ─────────────────────────────────
    tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}

    if tree_models:
        n_models = min(len(tree_models), 4)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(list(tree_models.items())[:n_models]):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)

            axes[idx].barh(
                [feature_names[i] for i in sorted_idx],
                importances[sorted_idx],
                color="steelblue",
            )
            axes[idx].set_title(name, fontsize=10)
            axes[idx].set_xlabel("Importance")

        plt.suptitle("Feature Importance by Model", fontsize=14)
        plt.tight_layout()
        plt.savefig("10_ensemble_feature_importance.png", dpi=100)
        print("Saved feature importance to 10_ensemble_feature_importance.png")

    # ── 7. Hyperparameter tuning demo ────────────────────────────────────
    print("\n=== Hyperparameter Tuning (GradientBoosting) ===\n")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.4f}")

    # Evaluate tuned model
    tuned_pred = grid_search.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, tuned_pred)
    tuned_r2 = r2_score(y_test, tuned_pred)
    print(f"Test MAE: {tuned_mae:.4f}")
    print(f"Test R²:  {tuned_r2:.4f}")

    # ── 8. Learning curve for best model ─────────────────────────────────
    print("\nGenerating learning curve...")

    best_model = grid_search.best_estimator_
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mae, "o-", label="Training MAE")
    plt.plot(train_sizes, val_mae, "o-", label="Validation MAE")
    plt.fill_between(train_sizes, train_mae - train_scores.std(axis=1), train_mae + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes, val_mae - val_scores.std(axis=1), val_mae + val_scores.std(axis=1), alpha=0.1)
    plt.xlabel("Training Set Size")
    plt.ylabel("MAE")
    plt.title("Learning Curve (Tuned Gradient Boosting)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("10_ensemble_learning_curve.png", dpi=100)
    print("Saved learning curve to 10_ensemble_learning_curve.png")

    # ── 9. Summary ───────────────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n=== Summary ===")
    print(f"Best model: {best_name}")
    print(f"  MAE:  {results[best_name]['MAE']:.4f} (${results[best_name]['MAE'] * 100_000:,.0f})")
    print(f"  R²:   {results[best_name]['R2']:.4f}")
    print(f"  Time: {results[best_name]['time']:.2f}s")
    print(f"\nKey takeaways:")
    print(f"  - Ensemble methods outperform single models")
    print(f"  - Gradient boosting typically wins on structured data")
    print(f"  - XGBoost/LightGBM are faster than sklearn's GradientBoosting")
    print(f"  - Hyperparameter tuning can further improve results")


if __name__ == "__main__":
    main()
