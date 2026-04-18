"""
Project 8: Time Series Forecasting
====================================
Forecast temperature using historical weather patterns.

What you'll learn:
- Time series data structure and stationarity
- Lag features, rolling statistics, and seasonal decomposition
- Training models on sequential data (no random splits!)
- ARIMA-style features with scikit-learn
- Comparing Linear Regression, Random Forest, and gradient boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def create_weather_dataset(n_years=5):
    """Generate synthetic daily temperature data with trend, seasonality, and noise."""
    np.random.seed(42)
    n_days = n_years * 365

    dates = pd.date_range(start="2019-01-01", periods=n_days, freq="D")

    # Components
    day_of_year = np.arange(n_days) % 365
    seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)  # yearly cycle
    trend = 0.003 * np.arange(n_days)  # slight warming trend
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # weekly pattern
    noise = np.random.normal(0, 3, n_days)

    # Autocorrelation: today's temp depends on yesterday's
    temp = np.zeros(n_days)
    temp[0] = 20 + seasonal[0] + noise[0]
    for i in range(1, n_days):
        temp[i] = 0.7 * temp[i - 1] + 0.3 * (20 + seasonal[i] + trend[i] + weekly[i]) + noise[i] * 0.5

    # Additional features
    humidity = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) + np.random.normal(0, 8, n_days)
    wind_speed = np.abs(8 + 4 * np.sin(2 * np.pi * day_of_year / 365 + 0.5) + np.random.normal(0, 3, n_days))
    pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n_days)

    df = pd.DataFrame({
        "date": dates,
        "temperature": temp,
        "humidity": humidity.clip(10, 100),
        "wind_speed": wind_speed.clip(0, 40),
        "pressure": pressure.clip(980, 1050),
    })
    df.set_index("date", inplace=True)
    return df


def add_time_features(df, target_col="temperature"):
    """Create time-based and lag features for forecasting."""
    out = df.copy()

    # Calendar features
    out["day_of_year"] = out.index.dayofyear
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["quarter"] = out.index.quarter
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    # Cyclical encoding (so Dec 31 is close to Jan 1)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)

    # Lag features (past temperatures)
    for lag in [1, 2, 3, 7, 14, 30]:
        out[f"temp_lag_{lag}"] = out[target_col].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        out[f"temp_roll_mean_{window}"] = out[target_col].shift(1).rolling(window).mean()
        out[f"temp_roll_std_{window}"] = out[target_col].shift(1).rolling(window).std()

    # Rolling stats for other features
    out["humidity_roll_7"] = out["humidity"].shift(1).rolling(7).mean()
    out["wind_roll_7"] = out["wind_speed"].shift(1).rolling(7).mean()

    # Temperature change features
    out["temp_diff_1"] = out[target_col].shift(1).diff()
    out["temp_diff_7"] = out[target_col].shift(1) - out[target_col].shift(8)

    return out


def main():
    # ── 1. Generate the dataset ──────────────────────────────────────────
    df = create_weather_dataset(n_years=5)

    print("=== Time Series Forecasting: Weather ===")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total days: {len(df)}")
    print(f"\nTemperature stats:")
    print(f"  Min:  {df['temperature'].min():.1f}°C")
    print(f"  Mean: {df['temperature'].mean():.1f}°C")
    print(f"  Max:  {df['temperature'].max():.1f}°C\n")

    # ── 2. Visualize the time series ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Full time series
    axes[0].plot(df.index, df["temperature"], linewidth=0.5, alpha=0.8)
    axes[0].set_title("Daily Temperature Over 5 Years")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)

    # Monthly averages
    monthly = df["temperature"].resample("ME").agg(["mean", "min", "max"])
    axes[1].fill_between(monthly.index, monthly["min"], monthly["max"], alpha=0.3, label="Min-Max Range")
    axes[1].plot(monthly.index, monthly["mean"], "b-", linewidth=2, label="Monthly Mean")
    axes[1].set_title("Monthly Temperature Summary")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Autocorrelation
    lags = range(1, 61)
    autocorr = [df["temperature"].autocorrelation(lag=l) for l in lags]
    axes[2].bar(lags, autocorr, color="steelblue", alpha=0.7)
    axes[2].set_xlabel("Lag (days)")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_title("Temperature Autocorrelation")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("08_timeseries_exploration.png", dpi=100)
    print("Saved exploration plot to 08_timeseries_exploration.png")

    # ── 3. Feature engineering ───────────────────────────────────────────
    df_feat = add_time_features(df, target_col="temperature")
    df_feat.dropna(inplace=True)

    feature_cols = [c for c in df_feat.columns if c != "temperature"]
    print(f"Features created: {len(feature_cols)}")
    print(f"Samples after dropping NaN: {len(df_feat)}\n")

    # ── 4. Time-based train/test split (NO random shuffle!) ─────────────
    split_date = "2023-01-01"
    train = df_feat[df_feat.index < split_date]
    test = df_feat[df_feat.index >= split_date]

    X_train = train[feature_cols].values
    y_train = train["temperature"].values
    X_test = test[feature_cols].values
    y_test = test["temperature"].values

    print(f"Training period: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Testing period:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 5. Train and compare models ──────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    print("=== Model Comparison ===\n")
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": y_pred}

        print(f"{name}:")
        print(f"  MAE:  {mae:.2f}°C")
        print(f"  RMSE: {rmse:.2f}°C")
        print(f"  R²:   {r2:.4f}\n")

    # ── 6. Visualize predictions ─────────────────────────────────────────
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)))

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(test.index, y_test, label="Actual", linewidth=1, alpha=0.8)
        ax.plot(test.index, res["predictions"], label="Predicted", linewidth=1, alpha=0.8)
        ax.set_title(f"{name} (MAE: {res['MAE']:.2f}°C, R²: {res['R2']:.4f})")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Temperature Forecasting: Model Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("08_timeseries_predictions.png", dpi=100)
    print("Saved predictions plot to 08_timeseries_predictions.png")

    # ── 7. Feature importance ────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["MAE"])
    best_model = models[best_name]

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        top_n = 15
        top_idx = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feature_cols[i] for i in top_idx],
            importances[top_idx],
            color="steelblue",
        )
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Features ({best_name})")
        plt.tight_layout()
        plt.savefig("08_timeseries_feature_importance.png", dpi=100)
        print("Saved feature importance to 08_timeseries_feature_importance.png")

        print(f"\nTop 10 features ({best_name}):")
        for i in reversed(top_idx[-10:]):
            print(f"  {feature_cols[i]:25s}: {importances[i]:.4f}")

    # ── 8. Forecast error analysis ───────────────────────────────────────
    best_pred = results[best_name]["predictions"]
    errors = y_test - best_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_title("Error Distribution")
    axes[0].set_xlabel("Error (°C)")
    axes[0].axvline(x=0, color="red", linestyle="--")

    axes[1].scatter(best_pred, errors, alpha=0.3, s=10)
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted Temperature")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Residual Plot")

    # Error by month
    error_df = pd.DataFrame({"error": np.abs(errors), "month": test.index.month})
    error_df.groupby("month")["error"].mean().plot(kind="bar", ax=axes[2], color="steelblue", alpha=0.7)
    axes[2].set_title("Mean Absolute Error by Month")
    axes[2].set_ylabel("MAE (°C)")
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("08_timeseries_error_analysis.png", dpi=100)
    print("Saved error analysis to 08_timeseries_error_analysis.png")

    print(f"\n=== Best Model: {best_name} ===")
    print(f"Average error: {np.mean(np.abs(errors)):.2f}°C")
    print(f"95% of predictions within: {np.percentile(np.abs(errors), 95):.2f}°C")


if __name__ == "__main__":
    main()
