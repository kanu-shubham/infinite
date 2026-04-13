"""
Project 9: Credit Card Fraud Detection
========================================
Detect fraudulent transactions in a highly imbalanced dataset.

What you'll learn:
- Handling imbalanced classes (fraud is rare: ~1-2%)
- SMOTE oversampling and class weight balancing
- Precision vs Recall tradeoff (catching fraud vs false alarms)
- Threshold tuning for business impact
- ROC and Precision-Recall curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc, average_precision_score,
    f1_score, precision_score, recall_score,
)


def create_fraud_dataset(n_samples=10000, fraud_ratio=0.02):
    """Create a synthetic credit card fraud dataset."""
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        "amount": np.random.lognormal(3.5, 1.2, n_legit).clip(1, 5000),
        "time_hour": np.random.choice(range(24), n_legit, p=[
            0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06,
            0.07, 0.07, 0.07, 0.07, 0.08, 0.07, 0.06, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02, 0.015,
        ]),
        "distance_from_home": np.random.exponential(15, n_legit).clip(0, 200),
        "distance_from_last_txn": np.random.exponential(5, n_legit).clip(0, 100),
        "ratio_to_median_purchase": np.random.lognormal(0, 0.5, n_legit).clip(0.1, 10),
        "repeat_retailer": np.random.choice([0, 1], n_legit, p=[0.2, 0.8]),
        "used_chip": np.random.choice([0, 1], n_legit, p=[0.3, 0.7]),
        "used_pin": np.random.choice([0, 1], n_legit, p=[0.4, 0.6]),
        "online_order": np.random.choice([0, 1], n_legit, p=[0.6, 0.4]),
        "txn_velocity_1h": np.random.poisson(1, n_legit),
        "txn_velocity_24h": np.random.poisson(4, n_legit),
        "is_fraud": 0,
    })

    # Fraudulent transactions (different patterns)
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(5.5, 1.5, n_fraud).clip(50, 15000),
        "time_hour": np.random.choice(range(24), n_fraud, p=[
            0.06, 0.07, 0.08, 0.08, 0.06, 0.04, 0.03, 0.02,
            0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06,
        ]),
        "distance_from_home": np.random.exponential(60, n_fraud).clip(5, 500),
        "distance_from_last_txn": np.random.exponential(40, n_fraud).clip(5, 300),
        "ratio_to_median_purchase": np.random.lognormal(1.5, 0.8, n_fraud).clip(1, 50),
        "repeat_retailer": np.random.choice([0, 1], n_fraud, p=[0.7, 0.3]),
        "used_chip": np.random.choice([0, 1], n_fraud, p=[0.7, 0.3]),
        "used_pin": np.random.choice([0, 1], n_fraud, p=[0.8, 0.2]),
        "online_order": np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),
        "txn_velocity_1h": np.random.poisson(4, n_fraud),
        "txn_velocity_24h": np.random.poisson(10, n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    df = create_fraud_dataset(n_samples=10000, fraud_ratio=0.02)

    print("=== Credit Card Fraud Detection ===")
    print(f"Total transactions: {len(df):,}")
    print(f"Legitimate:         {(df['is_fraud'] == 0).sum():,} ({(df['is_fraud'] == 0).mean():.1%})")
    print(f"Fraudulent:         {(df['is_fraud'] == 1).sum():,} ({df['is_fraud'].mean():.1%})")
    print(f"\n{df.describe().round(2)}\n")

    # ── 2. Explore fraud patterns ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Amount distribution
    for label, color, name in [(0, "green", "Legit"), (1, "red", "Fraud")]:
        subset = df[df["is_fraud"] == label]["amount"]
        axes[0, 0].hist(subset, bins=50, alpha=0.6, label=name, color=color, density=True)
    axes[0, 0].set_title("Transaction Amount")
    axes[0, 0].set_xlabel("Amount ($)")
    axes[0, 0].legend()

    # Time distribution
    for label, color, name in [(0, "green", "Legit"), (1, "red", "Fraud")]:
        subset = df[df["is_fraud"] == label]["time_hour"]
        axes[0, 1].hist(subset, bins=24, alpha=0.6, label=name, color=color, density=True)
    axes[0, 1].set_title("Transaction Time")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].legend()

    # Distance from home
    for label, color, name in [(0, "green", "Legit"), (1, "red", "Fraud")]:
        subset = df[df["is_fraud"] == label]["distance_from_home"]
        axes[0, 2].hist(subset, bins=50, alpha=0.6, label=name, color=color, density=True)
    axes[0, 2].set_title("Distance from Home")
    axes[0, 2].legend()

    # Feature comparison
    features_to_compare = ["ratio_to_median_purchase", "txn_velocity_1h", "txn_velocity_24h"]
    for i, feat in enumerate(features_to_compare):
        means = df.groupby("is_fraud")[feat].mean()
        axes[1, i].bar(["Legit", "Fraud"], means.values, color=["green", "red"], alpha=0.7)
        axes[1, i].set_title(f"Mean {feat}")

    plt.suptitle("Fraud vs Legitimate Transaction Patterns", fontsize=14)
    plt.tight_layout()
    plt.savefig("09_fraud_exploration.png", dpi=100)
    print("Saved exploration to 09_fraud_exploration.png")

    # ── 3. Prepare data ──────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"Testing:  {len(X_test)} ({y_test.sum()} fraud)\n")

    # ── 4. The imbalance problem ─────────────────────────────────────────
    print("=== The Imbalance Problem ===\n")

    # Naive model: predict everything as legit
    naive_acc = 1 - y_test.mean()
    print(f"Naive 'all legit' accuracy: {naive_acc:.1%} -- looks great but catches ZERO fraud!\n")

    # ── 5. Train models with different strategies ────────────────────────
    models = {
        "Logistic (no balancing)": LogisticRegression(max_iter=1000, random_state=42),
        "Logistic (class_weight=balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
    }

    print("=== Model Comparison ===\n")
    print(f"{'Model':<36} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 75)

    results = {}
    for name, model in models.items():
        if "Gradient Boosting" in name:
            # Use sample weights for gradient boosting
            weight = np.where(y_train == 1, len(y_train) / (2 * y_train.sum()), len(y_train) / (2 * (len(y_train) - y_train.sum())))
            model.fit(X_train_scaled, y_train, sample_weight=weight)
        else:
            model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {"predictions": y_pred, "probabilities": y_prob, "f1": f1}

        print(f"{name:<36} {acc:>8.1%} {prec:>9.1%} {rec:>8.1%} {f1:>7.3f}")

    print()

    # ── 6. ROC and Precision-Recall curves ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, res in results.items():
        probs = res["probabilities"]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        axes[1].plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("09_fraud_curves.png", dpi=100)
    print("Saved ROC and PR curves to 09_fraud_curves.png")

    # ── 7. Threshold tuning ──────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_probs = results[best_name]["probabilities"]

    print(f"\n=== Threshold Tuning ({best_name}) ===\n")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Fraud Caught':>13} {'False Alarms':>13}")
    print("-" * 68)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1, best_threshold = 0, 0.5

    for t in thresholds:
        y_pred_t = (best_probs >= t).astype(int)
        prec = precision_score(y_test, y_pred_t, zero_division=0)
        rec = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        caught = y_pred_t[y_test == 1].sum()
        total_fraud = y_test.sum()
        false_alarms = y_pred_t[y_test == 0].sum()

        print(f"{t:>10.1f} {prec:>9.1%} {rec:>8.1%} {f1:>7.3f} {caught:>6}/{total_fraud:<6} {false_alarms:>13}")

        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"\nOptimal threshold: {best_threshold} (F1: {best_f1:.3f})")

    # ── 8. Final confusion matrix with best threshold ────────────────────
    y_final = (best_probs >= best_threshold).astype(int)

    print(f"\n=== Final Results (threshold={best_threshold}) ===")
    print(classification_report(y_test, y_final, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Fraud Detection Confusion Matrix (threshold={best_threshold})")
    plt.tight_layout()
    plt.savefig("09_fraud_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 09_fraud_confusion_matrix.png")

    # Business impact
    avg_fraud_amount = df[df["is_fraud"] == 1]["amount"].mean()
    caught_ratio = recall_score(y_test, y_final)
    print(f"\n=== Business Impact ===")
    print(f"Average fraud amount: ${avg_fraud_amount:,.2f}")
    print(f"Fraud detection rate: {caught_ratio:.1%}")
    print(f"Estimated savings per 10,000 transactions: ${avg_fraud_amount * y_test.sum() * caught_ratio:,.2f}")


if __name__ == "__main__":
    main()
