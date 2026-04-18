"""
Project 21: Support Vector Machines (SVM)
==========================================
SVM finds the widest possible margin between classes.
It's one of the most powerful and elegant algorithms in ML.

What you'll learn:
- The core idea: maximum margin classifier
- Support vectors — the only points that matter
- The kernel trick: classifying non-linearly separable data
- Linear, RBF, and Polynomial kernels
- C parameter: controlling margin width vs misclassification
- When to use SVM vs logistic regression vs tree models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_circles, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


def plot_decision_boundary(ax, model, X, y, title, resolution=300):
    """Visualize the decision boundary and support vectors."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=1)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k",
                         linewidth=0.5, s=40, zorder=3)

    # Highlight support vectors
    if hasattr(model, "support_vectors_"):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=150, facecolors="none", edgecolors="gold",
                   linewidth=2, zorder=4, label=f"Support vectors ({len(model.support_vectors_)})")
        ax.legend(fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return scatter


def main():
    print("=== Support Vector Machines (SVM) ===\n")

    # ── Part 1: The core intuition ────────────────────────────────────────
    print("── Core Idea ──\n")
    print("""
  Logistic Regression: finds ANY line that separates classes
  SVM:                 finds the line with the MAXIMUM MARGIN

  Margin = distance from the decision boundary to the nearest point
  Larger margin → better generalization → less overfitting

  Support vectors: the data points closest to the boundary.
  Only these points define the model. Remove any other point
  and the boundary doesn't change.
    """)

    # ── Part 2: Linear SVM — visualize margin ────────────────────────────
    print("── Part 2: Linear SVM — Maximum Margin ──\n")

    np.random.seed(42)
    X_lin, y_lin = make_classification(
        n_samples=100, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=42,
    )

    # Compare different C values (C controls margin width vs misclassification)
    C_values = [0.01, 1.0, 100.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, C in zip(axes, C_values):
        svm = SVC(kernel="linear", C=C)
        svm.fit(X_lin, y_lin)
        acc = accuracy_score(y_lin, svm.predict(X_lin))
        n_sv = len(svm.support_vectors_)
        plot_decision_boundary(ax, svm, X_lin, y_lin,
                               f"Linear SVM  C={C}\nAcc={acc:.0%}  SVs={n_sv}")

    plt.suptitle("Effect of C Parameter\n"
                 "Small C → wide margin (more misclassifications allowed)\n"
                 "Large C → narrow margin (fits training data more tightly)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("21_svm_linear_margin.png", dpi=100)
    print("Saved linear SVM plot to 21_svm_linear_margin.png")

    for C in C_values:
        svm = SVC(kernel="linear", C=C)
        svm.fit(X_lin, y_lin)
        print(f"  C={C:6.2f} | Support vectors: {len(svm.support_vectors_):3d} | Train acc: {accuracy_score(y_lin, svm.predict(X_lin)):.1%}")

    # ── Part 3: The Kernel Trick — non-linear data ────────────────────────
    print("\n── Part 3: The Kernel Trick ──\n")
    print("""
  Problem: what if data is NOT linearly separable?
  Solution: map data to a higher-dimensional space where it IS separable.

  The kernel trick computes this mapping implicitly (no actual transformation).

  Common kernels:
    Linear:  K(x,z) = x·z              (no transformation)
    RBF:     K(x,z) = exp(-γ||x-z||²)  (infinite-dimensional space)
    Poly:    K(x,z) = (x·z + r)^d      (polynomial features)
    """)

    # Datasets that need the kernel trick
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)
    X_moons,   y_moons   = make_moons(n_samples=200, noise=0.15, random_state=42)

    datasets = [
        (X_circles, y_circles, "Circles"),
        (X_moons,   y_moons,   "Moons"),
    ]
    kernels = [
        ("linear", {}),
        ("rbf",    {"gamma": "scale"}),
        ("poly",   {"degree": 3}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, (X_d, y_d, dname) in enumerate(datasets):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_d)

        for col, (kernel, kwargs) in enumerate(kernels):
            svm = SVC(kernel=kernel, C=1.0, **kwargs)
            svm.fit(X_s, y_d)
            acc = accuracy_score(y_d, svm.predict(X_s))
            plot_decision_boundary(
                axes[row, col], svm, X_s, y_d,
                f"{dname} — {kernel.upper()} kernel\nAcc={acc:.1%}",
            )

    plt.suptitle("Kernel Trick: Same Data, Different Kernels", fontsize=13)
    plt.tight_layout()
    plt.savefig("21_svm_kernels.png", dpi=100)
    print("Saved kernel comparison to 21_svm_kernels.png")

    # ── Part 4: Real dataset — Breast Cancer ─────────────────────────────
    print("\n── Part 4: Real Dataset — Breast Cancer Classification ──\n")

    cancer = load_breast_cancer()
    X_raw, y = cancer.data, cancer.target

    print(f"Samples:  {len(X_raw)}")
    print(f"Features: {X_raw.shape[1]}")
    print(f"Classes:  {list(cancer.target_names)}")
    print(f"Positive (malignant): {(y==0).sum()}, Negative (benign): {(y==1).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Compare kernels
    print(f"{'Kernel':<10} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10}")
    print("-" * 45)

    best_name, best_acc = None, 0
    best_model = None
    for kernel in ["linear", "rbf", "poly"]:
        svm = SVC(kernel=kernel, C=1.0, random_state=42)
        svm.fit(X_train_s, y_train)
        tr_acc = accuracy_score(y_train, svm.predict(X_train_s))
        te_acc = accuracy_score(y_test,  svm.predict(X_test_s))
        cv     = cross_val_score(svm, X_train_s, y_train, cv=5).mean()
        print(f"{kernel:<10} {tr_acc:>9.1%} {te_acc:>9.1%} {cv:>9.1%}")
        if te_acc > best_acc:
            best_acc, best_name, best_model = te_acc, kernel, svm

    # ── Part 5: Hyperparameter tuning ────────────────────────────────────
    print(f"\n── Part 5: Hyperparameter Tuning (Grid Search) ──\n")

    param_grid = {
        "C":     [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01],
    }
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_s, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV acc: {grid.best_score_:.2%}")

    tuned_pred = grid.predict(X_test_s)
    tuned_acc  = accuracy_score(y_test, tuned_pred)
    print(f"Test acc after tuning: {tuned_acc:.2%}\n")

    print("Classification Report (tuned RBF SVM):")
    print(classification_report(y_test, tuned_pred, target_names=cancer.target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, tuned_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"SVM (RBF, tuned) — {tuned_acc:.1%} accuracy")
    plt.tight_layout()
    plt.savefig("21_svm_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 21_svm_confusion_matrix.png")

    # ── Part 6: When to use SVM ──────────────────────────────────────────
    print("\n── When to Use SVM ──\n")
    print("  USE SVM when:")
    print("    ✓ High-dimensional data (text, genomics) — linear kernel")
    print("    ✓ Small to medium datasets (< 100k samples)")
    print("    ✓ Clear margin of separation exists")
    print("    ✓ Need a maximum-margin guarantee")
    print("\n  AVOID SVM when:")
    print("    ✗ Very large datasets (slow to train — O(n²) to O(n³))")
    print("    ✗ Need probability estimates (SVM gives no probabilities by default)")
    print("    ✗ Noisy data with many overlapping classes")
    print("    ✗ You want feature importance (SVM doesn't provide it easily)")


if __name__ == "__main__":
    main()
