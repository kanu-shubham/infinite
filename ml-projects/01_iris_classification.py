"""
Project 1: Iris Flower Classification
======================================
Classify iris flowers into 3 species (Setosa, Versicolor, Virginica)
based on sepal and petal measurements.

What you'll learn:
- Loading and exploring a dataset
- Train/test splitting
- Training a classifier (K-Nearest Neighbors)
- Evaluating accuracy with a confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("=== Iris Dataset ===")
    print(f"Samples: {len(df)}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Species: {list(iris.target_names)}")
    print(f"\nFirst 5 rows:\n{df.head()}\n")
    print(f"Class distribution:\n{df['species'].value_counts()}\n")

    # ── 2. Visualize the data ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Petal length vs petal width scatter
    for species in iris.target_names:
        subset = df[df["species"] == species]
        axes[0].scatter(
            subset["petal length (cm)"],
            subset["petal width (cm)"],
            label=species,
            alpha=0.7,
        )
    axes[0].set_xlabel("Petal Length (cm)")
    axes[0].set_ylabel("Petal Width (cm)")
    axes[0].set_title("Iris: Petal Length vs Width")
    axes[0].legend()

    # Feature distributions
    df.drop(columns="species").hist(ax=axes[1] if False else None, bins=20, figsize=(10, 6))
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig("01_iris_visualization.png", dpi=100, bbox_inches="tight")
    print("Saved visualization to 01_iris_visualization.png")

    # ── 3. Prepare the data ──────────────────────────────────────────────
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    # ── 4. Train the model ───────────────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.tight_layout()
    plt.savefig("01_iris_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 01_iris_confusion_matrix.png")

    # ── 6. Experiment: try different K values ────────────────────────────
    print("\n=== Experiment: Accuracy vs. K ===")
    k_values = range(1, 21)
    accuracies = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        accuracies.append(acc)
        print(f"  K={k:2d} -> Accuracy: {acc:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker="o")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN: Accuracy vs. K Value")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("01_iris_k_experiment.png", dpi=100)
    print("\nSaved K experiment plot to 01_iris_k_experiment.png")


if __name__ == "__main__":
    main()
