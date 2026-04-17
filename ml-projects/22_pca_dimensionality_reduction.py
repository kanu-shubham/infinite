"""
Project 22: PCA & Dimensionality Reduction
===========================================
Principal Component Analysis (PCA) finds the directions of maximum variance
in your data and projects it into fewer dimensions without losing much information.

What you'll learn:
- Why dimensionality reduction matters (curse of dimensionality)
- How PCA works: eigenvectors, eigenvalues, explained variance
- Scree plot: choosing the right number of components
- Visualizing high-dimensional data in 2D
- PCA for denoising and compression
- t-SNE: non-linear alternative to PCA for visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, load_breast_cancer, fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=== PCA & Dimensionality Reduction ===\n")

    # ── Part 1: The intuition ─────────────────────────────────────────────
    print("── Core Idea ──\n")
    print("""
  High-dimensional data has a problem: most of that space is empty.
  With 100 features, data points are incredibly sparse — models struggle.

  PCA solution:
    1. Find the direction of greatest variance         (1st principal component)
    2. Find the next direction (perpendicular to 1st)  (2nd principal component)
    3. Keep only the top K components that capture ~95% of variance

  Result: fewer features, less noise, faster training, better visualization.
    """)

    # ── Part 2: PCA from scratch on 2D data ──────────────────────────────
    print("── Part 2: PCA Visualized on 2D Data ──\n")

    np.random.seed(42)
    # Correlated 2D data (elongated cloud)
    mean = [0, 0]
    cov  = [[3, 2], [2, 2]]
    X_2d = np.random.multivariate_normal(mean, cov, 300)

    # Compute PCA manually to show the math
    X_centered = X_2d - X_2d.mean(axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"Eigenvalues:   {eigenvalues.round(3)}")
    print(f"Explained var: {(eigenvalues / eigenvalues.sum() * 100).round(1)} %")

    # Project onto 1st principal component
    X_pca1d = X_centered @ eigenvectors[:, :1]

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original data + principal components
    scale = 3
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.4, s=20)
    origin = X_2d.mean(axis=0)
    for i, (ev, color, label) in enumerate(zip(
        eigenvectors.T, ["red", "blue"],
        ["PC1 (max variance)", "PC2 (perpendicular)"]
    )):
        axes[0].annotate("", origin + scale * np.sqrt(eigenvalues[i]) * ev,
                         origin, arrowprops=dict(color=color, lw=2, arrowstyle="->"))
        axes[0].text(*(origin + scale * np.sqrt(eigenvalues[i]) * ev + [0.1, 0.1]),
                     label, color=color, fontsize=9)
    axes[0].set_title("Original Data + Principal Components")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # 1D projection
    reconstructed = X_pca1d @ eigenvectors[:, :1].T + X_2d.mean(axis=0)
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, s=15, label="Original")
    axes[1].scatter(reconstructed[:, 0], reconstructed[:, 1],
                    alpha=0.6, s=15, c="red", label="Projected to PC1")
    for i in range(0, len(X_2d), 10):
        axes[1].plot([X_2d[i, 0], reconstructed[i, 0]],
                     [X_2d[i, 1], reconstructed[i, 1]], "gray", alpha=0.3, lw=0.5)
    axes[1].set_title("Projection onto PC1\n(2D → 1D, information loss shown)")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # Variance explained
    axes[2].bar(["PC1", "PC2"],
                eigenvalues / eigenvalues.sum() * 100,
                color=["red", "blue"], alpha=0.7)
    axes[2].set_ylabel("Explained Variance (%)")
    axes[2].set_title("Variance Explained by Each Component")
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(eigenvalues / eigenvalues.sum() * 100):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig("22_pca_2d_demo.png", dpi=100)
    print("Saved 2D demo to 22_pca_2d_demo.png\n")

    # ── Part 3: Scree plot — choosing K ──────────────────────────────────
    print("── Part 3: Digits Dataset — Choosing K Components ──\n")

    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target

    print(f"Original shape: {X_digits.shape}  ({X_digits.shape[1]} features = 8×8 pixels)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_digits)

    # Full PCA to see all components
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Find components needed for 90%, 95%, 99%
    for target in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cumulative, target) + 1
        print(f"  Components for {target:.0%} variance: {k}")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(range(1, len(explained) + 1), explained * 100, "b-o", markersize=3)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot\n(look for the 'elbow')")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=20, color="red", linestyle="--", label="K=20")
    axes[0].legend()

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100, "r-")
    axes[1].axhline(y=90, color="orange", linestyle="--", label="90%")
    axes[1].axhline(y=95, color="green",  linestyle="--", label="95%")
    axes[1].axhline(y=99, color="blue",   linestyle="--", label="99%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Digits Dataset: How Many Components Do We Need?", fontsize=12)
    plt.tight_layout()
    plt.savefig("22_pca_scree_plot.png", dpi=100)
    print("Saved scree plot to 22_pca_scree_plot.png")

    # ── Part 4: PCA for classification — speed vs accuracy ───────────────
    print("\n── Part 4: PCA for Classification — Speed vs Accuracy ──\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_digits,
                                               test_size=0.2, random_state=42)

    print(f"{'Components':>12} {'Dim':>6} {'Variance':>10} {'Accuracy':>10}")
    print("-" * 42)

    for k in [2, 10, 20, 30, 40, 64]:
        pca_k = PCA(n_components=k)
        X_tr_k = pca_k.fit_transform(X_tr)
        X_te_k = pca_k.transform(X_te)

        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr_k, y_tr)
        acc  = accuracy_score(y_te, clf.predict(X_te_k))
        var  = pca_k.explained_variance_ratio_.sum()
        print(f"{k:>12} {k:>6} {var:>9.1%} {acc:>9.1%}")

    # ── Part 5: Visualize in 2D ───────────────────────────────────────────
    print("\n── Part 5: 2D Visualization of 64-Dimensional Data ──\n")

    pca_2d = PCA(n_components=2)
    X_2d_proj = pca_2d.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA 2D scatter
    scatter = axes[0].scatter(X_2d_proj[:, 0], X_2d_proj[:, 1],
                              c=y_digits, cmap="tab10", alpha=0.6, s=15)
    plt.colorbar(scatter, ax=axes[0], label="Digit")
    axes[0].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].set_title("PCA: 64D → 2D\n(linear projection)")

    # t-SNE 2D scatter (better clustering, non-linear)
    print("Computing t-SNE (this takes ~30 seconds)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                               c=y_digits, cmap="tab10", alpha=0.6, s=15)
    plt.colorbar(scatter2, ax=axes[1], label="Digit")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].set_title("t-SNE: 64D → 2D\n(non-linear, better cluster separation)")

    plt.suptitle("PCA vs t-SNE: Visualizing 10 Digit Classes in 2D", fontsize=13)
    plt.tight_layout()
    plt.savefig("22_pca_vs_tsne.png", dpi=100)
    print("Saved PCA vs t-SNE to 22_pca_vs_tsne.png")

    # ── Part 6: PCA for image compression / denoising ────────────────────
    print("\n── Part 6: PCA for Image Compression ──\n")

    pca_components = [2, 5, 10, 20, 40, 64]
    sample_images  = X_digits[:5]

    fig, axes = plt.subplots(5, len(pca_components) + 1, figsize=(16, 10))

    for row in range(5):
        axes[row, 0].imshow(sample_images[row].reshape(8, 8), cmap="gray")
        axes[row, 0].set_title("Original" if row == 0 else "", fontsize=8)
        axes[row, 0].axis("off")

        for col, k in enumerate(pca_components):
            pca_k = PCA(n_components=k)
            pca_k.fit(X_scaled)
            compressed   = pca_k.transform(sample_images)
            reconstructed = pca_k.inverse_transform(compressed)
            reconstructed = scaler.inverse_transform(reconstructed)

            axes[row, col + 1].imshow(reconstructed[row].reshape(8, 8), cmap="gray")
            if row == 0:
                var = pca_k.explained_variance_ratio_.sum()
                axes[row, col + 1].set_title(f"K={k}\n{var:.0%}", fontsize=8)
            axes[row, col + 1].axis("off")

    plt.suptitle("Image Compression with PCA (K = number of components kept)", fontsize=12)
    plt.tight_layout()
    plt.savefig("22_pca_compression.png", dpi=100)
    print("Saved compression demo to 22_pca_compression.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Principal components — directions of maximum variance")
    print("  ✓ Explained variance ratio — how much info each component captures")
    print("  ✓ Scree plot — visual guide for choosing K")
    print("  ✓ PCA for speed — fewer features = faster training")
    print("  ✓ PCA for visualization — project any data to 2D/3D")
    print("  ✓ PCA for compression — encode and reconstruct images")
    print("  ✓ t-SNE — non-linear alternative for visualization only")
    print("\nRule of thumb: keep enough components for 95% explained variance")


if __name__ == "__main__":
    main()
