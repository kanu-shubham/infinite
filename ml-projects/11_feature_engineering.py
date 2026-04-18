"""
Project 11: Feature Engineering Masterclass
=============================================
Transform raw data into powerful features that dramatically improve models.

What you'll learn:
- Numerical transformations (log, binning, polynomial)
- Categorical encoding (one-hot, target, frequency)
- Feature interactions and domain-specific features
- Feature selection (correlation, importance, RFE)
- Measuring the impact of each technique
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, KBinsDiscretizer, PowerTransformer,
)
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def create_real_estate_dataset(n=3000):
    """Create a synthetic real estate dataset with rich features."""
    np.random.seed(42)

    # Core features
    sqft = np.random.lognormal(7.2, 0.4, n).clip(500, 10000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.15, 0.35, 0.25, 0.15, 0.05])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n, p=[0.6, 0.4]), 1, 5).astype(float)
    bathrooms += np.random.choice([0, 0.5], n, p=[0.7, 0.3])
    year_built = np.random.randint(1920, 2024, n)
    lot_size = sqft * np.random.uniform(1.5, 5, n)
    garage_cars = np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.35, 0.15])

    # Categorical
    neighborhood = np.random.choice(
        ["downtown", "suburb_a", "suburb_b", "rural", "waterfront", "historic"],
        n, p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.10],
    )
    condition = np.random.choice(["poor", "fair", "good", "excellent"], n, p=[0.10, 0.25, 0.40, 0.25])
    has_pool = np.random.choice([0, 1], n, p=[0.75, 0.25])
    has_fireplace = np.random.choice([0, 1], n, p=[0.60, 0.40])
    heating_type = np.random.choice(["gas", "electric", "oil", "solar"], n, p=[0.45, 0.30, 0.15, 0.10])

    # Derived target (price) with realistic relationships
    neighborhood_mult = {"downtown": 1.3, "suburb_a": 1.0, "suburb_b": 0.95, "rural": 0.7, "waterfront": 1.6, "historic": 1.2}
    condition_mult = {"poor": 0.7, "fair": 0.85, "good": 1.0, "excellent": 1.2}

    price = (
        50 * sqft
        + 15000 * bedrooms
        + 20000 * bathrooms
        + 200 * np.maximum(year_built - 1950, 0)
        + 5 * lot_size
        + 25000 * garage_cars
        + 40000 * has_pool
        + 15000 * has_fireplace
    )
    price *= np.array([neighborhood_mult[n] for n in neighborhood])
    price *= np.array([condition_mult[c] for c in condition])
    price *= np.random.normal(1.0, 0.1, n)  # noise
    price = price.clip(50000, 2000000)

    return pd.DataFrame({
        "sqft": sqft.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "lot_size": lot_size.astype(int),
        "garage_cars": garage_cars,
        "neighborhood": neighborhood,
        "condition": condition,
        "has_pool": has_pool,
        "has_fireplace": has_fireplace,
        "heating_type": heating_type,
        "price": price.astype(int),
    })


def main():
    # ── 1. Load data ─────────────────────────────────────────────────────
    df = create_real_estate_dataset(n=3000)

    print("=== Feature Engineering Masterclass ===")
    print(f"Samples: {len(df):,}")
    print(f"\nRaw features:\n{df.dtypes}\n")
    print(f"Price range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}\n")

    # ── 2. Baseline: raw features only ───────────────────────────────────
    print("=== Step 1: Baseline (Raw Numeric Features Only) ===\n")

    numeric_cols = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size",
                    "garage_cars", "has_pool", "has_fireplace"]

    X_base = df[numeric_cols].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    baseline_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    baseline_mae = -baseline_scores.mean()
    print(f"Baseline MAE (numeric only): ${baseline_mae:,.0f}\n")

    stage_results = {"1. Raw Numeric": baseline_mae}

    # ── 3. Categorical encoding ──────────────────────────────────────────
    print("=== Step 2: Add Categorical Encoding ===\n")

    df_enc = df.copy()

    # One-hot encoding for low-cardinality features
    df_enc = pd.get_dummies(df_enc, columns=["heating_type"], drop_first=True)

    # Ordinal encoding for condition
    condition_map = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
    df_enc["condition_ordinal"] = df_enc["condition"].map(condition_map)

    # Target encoding for neighborhood (using training data only to avoid leakage)
    train_idx, test_idx = train_test_split(range(len(df_enc)), test_size=0.2, random_state=42)
    neighborhood_means = df_enc.iloc[train_idx].groupby("neighborhood")["price"].mean()
    df_enc["neighborhood_target_enc"] = df_enc["neighborhood"].map(neighborhood_means)
    # Fill test neighborhoods not seen in training
    df_enc["neighborhood_target_enc"].fillna(df_enc.iloc[train_idx]["price"].mean(), inplace=True)

    # Frequency encoding
    neighborhood_freq = df_enc["neighborhood"].value_counts(normalize=True)
    df_enc["neighborhood_freq_enc"] = df_enc["neighborhood"].map(neighborhood_freq)

    cat_features = [c for c in df_enc.columns if c not in ["price", "neighborhood", "condition"]]
    X_cat = df_enc[cat_features].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_cat, y, test_size=0.2, random_state=42)

    scores = cross_val_score(model, X_train2, y_train2, cv=5, scoring="neg_mean_absolute_error")
    cat_mae = -scores.mean()
    print(f"With categorical encoding: ${cat_mae:,.0f}")
    print(f"Improvement: ${baseline_mae - cat_mae:,.0f} ({(baseline_mae - cat_mae) / baseline_mae:.1%})\n")

    stage_results["2. + Categorical"] = cat_mae

    # ── 4. Numerical transformations ─────────────────────────────────────
    print("=== Step 3: Add Numerical Transformations ===\n")

    df_trans = df_enc.copy()

    # Log transform skewed features
    for col in ["sqft", "lot_size", "price"]:
        if col in df_trans.columns:
            df_trans[f"{col}_log"] = np.log1p(df_trans[col])

    # Binning
    df_trans["year_bin"] = pd.cut(df_trans["year_built"], bins=[1900, 1950, 1970, 1990, 2005, 2025],
                                   labels=[0, 1, 2, 3, 4]).astype(float)

    # Sqrt transform
    df_trans["sqft_sqrt"] = np.sqrt(df_trans["sqft"])

    trans_features = [c for c in df_trans.columns if c not in ["price", "price_log", "neighborhood", "condition"]]
    X_trans = df_trans[trans_features].values
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_trans, y, test_size=0.2, random_state=42)

    scores = cross_val_score(model, X_train3, y_train3, cv=5, scoring="neg_mean_absolute_error")
    trans_mae = -scores.mean()
    print(f"With transformations: ${trans_mae:,.0f}")
    print(f"Improvement over baseline: ${baseline_mae - trans_mae:,.0f} ({(baseline_mae - trans_mae) / baseline_mae:.1%})\n")

    stage_results["3. + Transformations"] = trans_mae

    # ── 5. Feature interactions ──────────────────────────────────────────
    print("=== Step 4: Add Feature Interactions ===\n")

    df_inter = df_trans.copy()

    # Domain-specific interactions
    df_inter["price_per_sqft"] = df_inter["sqft_log"] / df_inter["bedrooms"].clip(1)
    df_inter["bath_bed_ratio"] = df_inter["bathrooms"] / df_inter["bedrooms"].clip(1)
    df_inter["total_rooms"] = df_inter["bedrooms"] + df_inter["bathrooms"]
    df_inter["house_age"] = 2024 - df_inter["year_built"]
    df_inter["age_sqft"] = df_inter["house_age"] * df_inter["sqft"]
    df_inter["luxury_score"] = df_inter["has_pool"] + df_inter["has_fireplace"] + (df_inter["garage_cars"] >= 2).astype(int)
    df_inter["sqft_per_bedroom"] = df_inter["sqft"] / df_inter["bedrooms"].clip(1)
    df_inter["lot_to_house_ratio"] = df_inter["lot_size"] / df_inter["sqft"].clip(1)
    df_inter["is_new_construction"] = (df_inter["year_built"] >= 2015).astype(int)
    df_inter["is_vintage"] = (df_inter["year_built"] <= 1950).astype(int)

    inter_features = [c for c in df_inter.columns if c not in ["price", "price_log", "neighborhood", "condition"]]
    X_inter = df_inter[inter_features].values
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X_inter, y, test_size=0.2, random_state=42)

    scores = cross_val_score(model, X_train4, y_train4, cv=5, scoring="neg_mean_absolute_error")
    inter_mae = -scores.mean()
    print(f"With interactions: ${inter_mae:,.0f}")
    print(f"Improvement over baseline: ${baseline_mae - inter_mae:,.0f} ({(baseline_mae - inter_mae) / baseline_mae:.1%})\n")

    stage_results["4. + Interactions"] = inter_mae

    # ── 6. Feature selection ─────────────────────────────────────────────
    print("=== Step 5: Feature Selection ===\n")

    # Train a Random Forest to get importances
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train4, y_train4)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": inter_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("Top 15 features by importance:")
    for _, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # Select top features
    top_features = importance_df.head(20)["feature"].tolist()
    top_feature_idx = [inter_features.index(f) for f in top_features]

    X_selected = X_inter[:, top_feature_idx]
    X_train5, X_test5, y_train5, y_test5 = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    scores = cross_val_score(model, X_train5, y_train5, cv=5, scoring="neg_mean_absolute_error")
    select_mae = -scores.mean()
    print(f"\nWith top 20 features: ${select_mae:,.0f}")
    print(f"Improvement over baseline: ${baseline_mae - select_mae:,.0f} ({(baseline_mae - select_mae) / baseline_mae:.1%})")
    print(f"(Reduced from {len(inter_features)} to {len(top_features)} features)\n")

    stage_results["5. + Selection"] = select_mae

    # ── 7. Visualize the journey ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # MAE improvement over stages
    stages = list(stage_results.keys())
    maes = list(stage_results.values())

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stages)))
    bars = axes[0].bar(range(len(stages)), maes, color=colors)
    axes[0].set_xticks(range(len(stages)))
    axes[0].set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("MAE ($)")
    axes[0].set_title("Feature Engineering Impact on MAE")
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"${val:,.0f}", ha="center", fontsize=9)

    # Feature importance
    top10 = importance_df.head(10)
    axes[1].barh(top10["feature"][::-1], top10["importance"][::-1], color="steelblue")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top 10 Features")

    plt.tight_layout()
    plt.savefig("11_feature_engineering_impact.png", dpi=100)
    print("Saved impact visualization to 11_feature_engineering_impact.png")

    # ── 8. Final evaluation on test set ──────────────────────────────────
    print("\n=== Final Test Set Evaluation ===\n")

    # Baseline on test
    model_base = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_base.fit(X_train, y_train)
    base_pred = model_base.predict(X_test)
    base_test_mae = mean_absolute_error(y_test, base_pred)
    base_test_r2 = r2_score(y_test, base_pred)

    # Engineered features on test
    model_eng = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_eng.fit(X_train5, y_train5)
    eng_pred = model_eng.predict(X_test5)
    eng_test_mae = mean_absolute_error(y_test5, eng_pred)
    eng_test_r2 = r2_score(y_test5, eng_pred)

    print(f"{'Metric':<10} {'Baseline':>15} {'Engineered':>15} {'Improvement':>15}")
    print("-" * 58)
    print(f"{'MAE':<10} ${base_test_mae:>13,.0f} ${eng_test_mae:>13,.0f} ${base_test_mae - eng_test_mae:>13,.0f}")
    print(f"{'R²':<10} {base_test_r2:>14.4f} {eng_test_r2:>14.4f} {eng_test_r2 - base_test_r2:>14.4f}")

    # Prediction scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pred, mae, r2, title in [
        (axes[0], base_pred, base_test_mae, base_test_r2, "Baseline (Raw Numeric)"),
        (axes[1], eng_pred, eng_test_mae, eng_test_r2, "Engineered Features"),
    ]:
        ax.scatter(y_test if title == "Baseline (Raw Numeric)" else y_test5, pred, alpha=0.3, s=10)
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2)
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title(f"{title}\nMAE: ${mae:,.0f} | R²: {r2:.4f}")

    plt.suptitle("Actual vs Predicted: Before & After Feature Engineering", fontsize=13)
    plt.tight_layout()
    plt.savefig("11_feature_engineering_predictions.png", dpi=100)
    print("\nSaved prediction comparison to 11_feature_engineering_predictions.png")


if __name__ == "__main__":
    main()
