"""
Project 7: Customer Segmentation
==================================
Group customers into distinct segments based on their purchasing behavior
using unsupervised learning (K-Means clustering).

What you'll learn:
- Unsupervised learning (no labels needed)
- K-Means clustering algorithm
- The Elbow Method to find optimal K
- Visualizing clusters in 2D with PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples


def create_customer_dataset():
    """Create a synthetic customer dataset with natural clusters."""
    np.random.seed(42)

    # Cluster 1: Budget shoppers (low spend, moderate frequency)
    n1 = 80
    c1 = pd.DataFrame({
        "annual_income": np.random.normal(25000, 5000, n1),
        "spending_score": np.random.normal(30, 10, n1),
        "purchase_frequency": np.random.normal(15, 5, n1),
        "avg_order_value": np.random.normal(25, 8, n1),
        "years_as_customer": np.random.normal(3, 1.5, n1),
    })

    # Cluster 2: Average customers (moderate everything)
    n2 = 100
    c2 = pd.DataFrame({
        "annual_income": np.random.normal(55000, 10000, n2),
        "spending_score": np.random.normal(50, 12, n2),
        "purchase_frequency": np.random.normal(25, 8, n2),
        "avg_order_value": np.random.normal(55, 15, n2),
        "years_as_customer": np.random.normal(5, 2, n2),
    })

    # Cluster 3: Premium shoppers (high income, high spend)
    n3 = 60
    c3 = pd.DataFrame({
        "annual_income": np.random.normal(95000, 12000, n3),
        "spending_score": np.random.normal(80, 8, n3),
        "purchase_frequency": np.random.normal(40, 10, n3),
        "avg_order_value": np.random.normal(120, 30, n3),
        "years_as_customer": np.random.normal(7, 2, n3),
    })

    # Cluster 4: High income, low engagement
    n4 = 60
    c4 = pd.DataFrame({
        "annual_income": np.random.normal(90000, 15000, n4),
        "spending_score": np.random.normal(20, 8, n4),
        "purchase_frequency": np.random.normal(8, 3, n4),
        "avg_order_value": np.random.normal(80, 25, n4),
        "years_as_customer": np.random.normal(2, 1, n4),
    })

    df = pd.concat([c1, c2, c3, c4], ignore_index=True)

    # Clip to reasonable values
    df["annual_income"] = df["annual_income"].clip(10000, 150000)
    df["spending_score"] = df["spending_score"].clip(1, 100)
    df["purchase_frequency"] = df["purchase_frequency"].clip(1, 60)
    df["avg_order_value"] = df["avg_order_value"].clip(5, 200)
    df["years_as_customer"] = df["years_as_customer"].clip(0.5, 15)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    df = create_customer_dataset()

    print("=== Customer Segmentation ===")
    print(f"Customers: {len(df)}")
    print(f"Features: {list(df.columns)}\n")
    print(f"Dataset overview:\n{df.describe().round(1)}\n")

    # ── 2. Explore the data ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat

    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=25, edgecolor="black", alpha=0.7, color="#4CAF50")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel(col)

    # Scatter: income vs spending
    axes[5].scatter(df["annual_income"], df["spending_score"], alpha=0.5, s=20)
    axes[5].set_xlabel("Annual Income")
    axes[5].set_ylabel("Spending Score")
    axes[5].set_title("Income vs Spending Score")

    plt.suptitle("Customer Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig("07_customer_distributions.png", dpi=100)
    print("Saved distributions to 07_customer_distributions.png")

    # ── 3. Scale the data ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ── 4. Elbow Method: find optimal K ──────────────────────────────────
    print("Running Elbow Method...")
    K_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil)
        print(f"  K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(K_range, inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(K_range, silhouette_scores, "ro-", linewidth=2)
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs. K")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_customer_elbow_method.png", dpi=100)
    print("\nSaved Elbow Method plot to 07_customer_elbow_method.png")

    # ── 5. Final clustering with K=4 ────────────────────────────────────
    optimal_k = 4
    print(f"\nUsing K={optimal_k} clusters\n")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # ── 6. Analyze clusters ──────────────────────────────────────────────
    print("=== Cluster Profiles ===\n")
    cluster_summary = df.groupby("cluster").agg(["mean", "count"]).round(1)

    feature_cols = [c for c in df.columns if c != "cluster"]
    for cluster_id in range(optimal_k):
        cluster_data = df[df["cluster"] == cluster_id]
        print(f"Cluster {cluster_id} ({len(cluster_data)} customers):")
        for col in feature_cols:
            mean_val = cluster_data[col].mean()
            print(f"  {col:25s}: {mean_val:.1f}")
        print()

    # Assign descriptive names based on characteristics
    cluster_means = df.groupby("cluster")[feature_cols].mean()
    print("Cluster Interpretation:")
    for cluster_id in range(optimal_k):
        row = cluster_means.loc[cluster_id]
        if row["spending_score"] > 65 and row["annual_income"] > 70000:
            label = "Premium Loyalists"
        elif row["spending_score"] < 35 and row["annual_income"] > 70000:
            label = "High-Income Passive"
        elif row["annual_income"] < 40000:
            label = "Budget Conscious"
        else:
            label = "Average Customers"
        print(f"  Cluster {cluster_id}: {label}")

    # ── 7. Visualize clusters with PCA ───────────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df["cluster"], cmap="viridis",
        alpha=0.6, s=50, edgecolors="white", linewidth=0.5,
    )

    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c="red", marker="X", s=200, edgecolors="black", linewidth=2,
        label="Centroids",
    )

    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"Customer Segments (K={optimal_k}, PCA Projection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("07_customer_clusters_pca.png", dpi=100)
    print("Saved PCA cluster plot to 07_customer_clusters_pca.png")

    # ── 8. Feature comparison radar-style chart ──────────────────────────
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(20, 5))

    for i, col in enumerate(feature_cols):
        means = df.groupby("cluster")[col].mean()
        axes[i].bar(means.index, means.values, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("Cluster")

    plt.suptitle("Feature Means by Cluster", fontsize=14)
    plt.tight_layout()
    plt.savefig("07_customer_cluster_profiles.png", dpi=100)
    print("Saved cluster profiles to 07_customer_cluster_profiles.png")


if __name__ == "__main__":
    main()
