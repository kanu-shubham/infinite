"""
Project 3: Titanic Survival Prediction
========================================
Predict passenger survival on the Titanic using demographic and ticket data.

What you'll learn:
- Handling missing data
- Encoding categorical features
- Feature engineering
- Logistic Regression and Decision Trees
- Cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_titanic_dataset():
    """Create a synthetic Titanic-like dataset (no download needed)."""
    np.random.seed(42)
    n = 891

    pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.clip(np.random.normal(30, 14, n), 0.5, 80)
    age[np.random.choice(n, size=177, replace=False)] = np.nan
    sibsp = np.random.choice(range(6), size=n, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01])
    parch = np.random.choice(range(5), size=n, p=[0.76, 0.12, 0.08, 0.02, 0.02])
    fare = np.where(pclass == 1, np.random.exponential(60, n) + 30,
           np.where(pclass == 2, np.random.exponential(20, n) + 10,
                    np.random.exponential(8, n) + 5))
    embarked = np.random.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
    embarked[np.random.choice(n, size=2, replace=False)] = np.nan

    survival_prob = (
        0.2
        + 0.3 * (np.array(sex) == "female").astype(float)
        + 0.15 * (pclass == 1).astype(float)
        + 0.05 * (pclass == 2).astype(float)
        - 0.1 * (np.nan_to_num(age, nan=30) > 50).astype(float)
        + 0.05 * (np.nan_to_num(age, nan=30) < 10).astype(float)
    )
    survival_prob = np.clip(survival_prob, 0.05, 0.95)
    survived = (np.random.random(n) < survival_prob).astype(int)

    return pd.DataFrame({
        "Survived": survived,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    })


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    df = create_titanic_dataset()

    print("=== Titanic Survival Dataset ===")
    print(f"Samples: {len(df)}")
    print(f"\nFirst 5 rows:\n{df.head()}\n")
    print(f"Survival rate: {df['Survived'].mean():.1%}")
    print(f"\nMissing values:\n{df.isnull().sum()}\n")

    # ── 2. Explore the data ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Survival by sex
    df.groupby("Sex")["Survived"].mean().plot(kind="bar", ax=axes[0, 0], color=["#ff7f7f", "#7fbfff"])
    axes[0, 0].set_title("Survival Rate by Sex")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].tick_params(axis="x", rotation=0)

    # Survival by class
    df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[0, 1], color=["#90EE90", "#FFD700", "#FFA07A"])
    axes[0, 1].set_title("Survival Rate by Class")
    axes[0, 1].set_ylabel("Survival Rate")
    axes[0, 1].tick_params(axis="x", rotation=0)

    # Age distribution
    df[df["Survived"] == 1]["Age"].hist(ax=axes[1, 0], bins=30, alpha=0.7, label="Survived", color="green")
    df[df["Survived"] == 0]["Age"].hist(ax=axes[1, 0], bins=30, alpha=0.7, label="Died", color="red")
    axes[1, 0].set_title("Age Distribution by Survival")
    axes[1, 0].legend()

    # Fare distribution
    df[df["Survived"] == 1]["Fare"].hist(ax=axes[1, 1], bins=30, alpha=0.7, label="Survived", color="green")
    df[df["Survived"] == 0]["Fare"].hist(ax=axes[1, 1], bins=30, alpha=0.7, label="Died", color="red")
    axes[1, 1].set_title("Fare Distribution by Survival")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("03_titanic_exploration.png", dpi=100)
    print("Saved exploration plots to 03_titanic_exploration.png")

    # ── 3. Feature engineering ───────────────────────────────────────────
    df_processed = df.copy()

    # Fill missing Age with median per class
    for pclass in [1, 2, 3]:
        median_age = df_processed[df_processed["Pclass"] == pclass]["Age"].median()
        mask = (df_processed["Pclass"] == pclass) & (df_processed["Age"].isnull())
        df_processed.loc[mask, "Age"] = median_age

    # Fill missing Embarked with mode
    df_processed["Embarked"].fillna(df_processed["Embarked"].mode()[0], inplace=True)

    # Create new features
    df_processed["FamilySize"] = df_processed["SibSp"] + df_processed["Parch"] + 1
    df_processed["IsAlone"] = (df_processed["FamilySize"] == 1).astype(int)
    df_processed["AgeGroup"] = pd.cut(df_processed["Age"], bins=[0, 12, 18, 35, 60, 100],
                                       labels=["Child", "Teen", "Adult", "Middle", "Senior"])

    # Encode categorical variables
    df_processed["Sex"] = LabelEncoder().fit_transform(df_processed["Sex"])
    df_processed = pd.get_dummies(df_processed, columns=["Embarked", "AgeGroup"], drop_first=True)

    print("\nEngineered features:")
    print(f"  FamilySize range: {df_processed['FamilySize'].min()}-{df_processed['FamilySize'].max()}")
    print(f"  Alone passengers: {df_processed['IsAlone'].sum()}")
    print(f"  Final feature count: {df_processed.shape[1] - 1}")

    # ── 4. Prepare the data ──────────────────────────────────────────────
    feature_cols = [c for c in df_processed.columns if c != "Survived"]
    X = df_processed[feature_cols].values
    y = df_processed["Survived"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    # ── 5. Train and compare models ──────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print("=== Model Comparison ===\n")
    best_name, best_acc = None, 0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        print(f"{name}:")
        print(f"  Test Accuracy:    {acc:.2%}")
        print(f"  CV Mean Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        print()

        if acc > best_acc:
            best_name, best_acc = name, acc

    print(f"Best model: {best_name} ({best_acc:.2%})\n")

    # ── 6. Detailed report for best model ────────────────────────────────
    best_model = models[best_name]
    y_pred = best_model.predict(X_test_scaled)

    print(f"=== {best_name} Detailed Report ===")
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Died", "Survived"],
                yticklabels=["Died", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{best_name} - Confusion Matrix ({best_acc:.2%})")
    plt.tight_layout()
    plt.savefig("03_titanic_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 03_titanic_confusion_matrix.png")


if __name__ == "__main__":
    main()
