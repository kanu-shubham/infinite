"""
Project 2: House Price Prediction
==================================
Predict house prices using the California Housing dataset.

What you'll learn:
- Regression (predicting continuous values)
- Feature correlation analysis
- Linear Regression vs. Random Forest
- Evaluating with MAE, RMSE, and R-squared
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target  # in $100,000s

    print("=== California Housing Dataset ===")
    print(f"Samples: {len(df)}")
    print(f"Features: {list(housing.feature_names)}")
    print(f"\nFirst 5 rows:\n{df.head()}\n")
    print(f"Target stats (Median House Value in $100k):")
    print(f"  Min:  ${df['MedHouseVal'].min() * 100_000:,.0f}")
    print(f"  Mean: ${df['MedHouseVal'].mean() * 100_000:,.0f}")
    print(f"  Max:  ${df['MedHouseVal'].max() * 100_000:,.0f}\n")

    # ── 2. Explore feature correlations ──────────────────────────────────
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("02_house_correlation.png", dpi=100)
    print("Saved correlation matrix to 02_house_correlation.png")

    # Top correlations with target
    target_corr = correlation["MedHouseVal"].drop("MedHouseVal").sort_values(ascending=False)
    print("\nCorrelation with house price:")
    for feat, corr in target_corr.items():
        print(f"  {feat:15s}: {corr:+.3f}")

    # ── 3. Prepare the data ──────────────────────────────────────────────
    X = df.drop(columns="MedHouseVal").values
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    # ── 4. Train models ──────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": y_pred}

        print(f"  MAE:  ${mae * 100_000:,.0f}")
        print(f"  RMSE: ${rmse * 100_000:,.0f}")
        print(f"  R²:   {r2:.4f}\n")

    # ── 5. Compare models visually ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        ax.scatter(y_test, res["predictions"], alpha=0.3, s=10)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            linewidth=2,
        )
        ax.set_xlabel("Actual Price ($100k)")
        ax.set_ylabel("Predicted Price ($100k)")
        ax.set_title(f"{name}\nR² = {res['R2']:.4f}")

    plt.suptitle("Actual vs Predicted House Prices", fontsize=14)
    plt.tight_layout()
    plt.savefig("02_house_predictions.png", dpi=100)
    print("Saved prediction comparison to 02_house_predictions.png")

    # ── 6. Feature importance (Random Forest) ────────────────────────────
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nRandom Forest Feature Importances:")
    for i in indices:
        print(f"  {housing.feature_names[i]:15s}: {importances[i]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [housing.feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("02_house_feature_importance.png", dpi=100)
    print("Saved feature importance plot to 02_house_feature_importance.png")


if __name__ == "__main__":
    main()
