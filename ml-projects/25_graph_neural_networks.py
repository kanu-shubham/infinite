"""
Project 25: Graph Neural Networks (GNNs)
==========================================
Graphs are everywhere: social networks, molecules, fraud detection,
knowledge graphs, citation networks. GNNs learn from graph structure.

What you'll learn:
- Graph representation: nodes, edges, adjacency matrix
- Message passing: each node aggregates info from its neighbors
- Graph Convolutional Network (GCN) from scratch
- Node classification: label each node (e.g., spam user detection)
- Link prediction: will these two nodes connect? (friend recommendation)
- Graph classification: is this molecule toxic?
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Graph data structure ──────────────────────────────────────────────────

class Graph:
    """Simple graph container."""
    def __init__(self, n_nodes, edge_index, node_features, labels=None):
        self.n_nodes       = n_nodes
        self.edge_index    = edge_index      # (2, n_edges) — [source, target]
        self.node_features = node_features   # (n_nodes, n_features)
        self.labels        = labels          # (n_nodes,) or (1,) for graph-level

    def to(self, device):
        self.edge_index    = self.edge_index.to(device)
        self.node_features = self.node_features.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self

    def get_adjacency(self):
        """Build normalized adjacency matrix with self-loops: D^{-1/2}(A+I)D^{-1/2}."""
        n = self.n_nodes
        A = torch.zeros(n, n)
        src, dst = self.edge_index
        A[src, dst] = 1
        A[dst, src] = 1   # undirected
        A += torch.eye(n)  # self-loops: each node also aggregates itself

        # Degree normalization
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(D.pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt


# ── GCN Layer from scratch ────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    H_new = σ( D^{-1/2} A D^{-1/2}  H  W )
              ↑ normalized adjacency  ↑  ↑
              (with self-loops)      features  learnable weights

    Intuition: each node's new representation = weighted average of
    its neighbors' old representations, linearly transformed.

    This is the graph analog of a convolutional layer:
      CNN:  slide a filter over local pixel neighborhoods
      GCN:  aggregate over graph neighborhoods (variable size)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, A_norm):
        """
        H:      (n_nodes, in_features)
        A_norm: (n_nodes, n_nodes) — normalized adjacency
        """
        return self.W(A_norm @ H)   # aggregate neighbors, then transform


class GCN(nn.Module):
    """
    Two-layer GCN for node classification.

    Input node features → GCN layer → ReLU → Dropout → GCN layer → Softmax
    """
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.conv1   = GCNLayer(in_features, hidden_dim)
        self.conv2   = GCNLayer(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A_norm):
        H = F.relu(self.conv1(H, A_norm))
        H = self.dropout(H)
        H = self.conv2(H, A_norm)
        return F.log_softmax(H, dim=1)


# ── GraphSAGE (more flexible message passing) ─────────────────────────────

class SAGELayer(nn.Module):
    """
    GraphSAGE layer (Hamilton et al., 2017).

    Instead of averaging ALL neighbors (GCN), GraphSAGE:
    1. Samples a fixed number of neighbors (scalable to large graphs)
    2. Concatenates [self, mean_of_neighbors] → linear layer

    Used in Pinterest's recommendation system (billions of nodes).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, H, A_norm):
        # Neighbor aggregation: mean of neighbors
        neigh_agg = A_norm @ H
        return self.W_self(H) + self.W_neigh(neigh_agg) + self.bias


# ── Synthetic graph datasets ──────────────────────────────────────────────

def create_karate_like_graph():
    """
    Create a synthetic social network with two communities.
    Task: predict which community each node belongs to (node classification).
    """
    np.random.seed(42)
    n_nodes = 80
    n_comm_a = 40

    edges = []
    # Dense connections within communities
    for i in range(n_comm_a):
        for j in range(i + 1, n_comm_a):
            if np.random.rand() < 0.15:
                edges.append([i, j])
    for i in range(n_comm_a, n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < 0.15:
                edges.append([i, j])
    # Sparse connections between communities
    for i in range(n_comm_a):
        for j in range(n_comm_a, n_nodes):
            if np.random.rand() < 0.02:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Node features: community membership + noise
    features = torch.zeros(n_nodes, 8)
    features[:n_comm_a,  :4] = torch.randn(n_comm_a, 4) + 1
    features[n_comm_a:, 4:]  = torch.randn(n_nodes - n_comm_a, 4) + 1
    features += torch.randn(n_nodes, 8) * 0.5

    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[n_comm_a:] = 1

    return Graph(n_nodes, edge_index, features, labels)


def create_fraud_graph():
    """
    Credit card fraud as a graph: nodes=transactions, edges=shared cards/merchants.
    Task: detect fraudulent transaction nodes.
    """
    np.random.seed(123)
    n_nodes  = 200
    n_fraud  = 20

    edges = []
    for i in range(n_nodes):
        n_neighbors = np.random.poisson(3)
        for _ in range(n_neighbors):
            j = np.random.randint(0, n_nodes)
            if j != i:
                edges.append([i, j])

    # Fraudulent nodes tend to cluster together
    fraud_nodes = np.random.choice(n_nodes, n_fraud, replace=False)
    for i in fraud_nodes:
        for j in fraud_nodes:
            if i != j and np.random.rand() < 0.4:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)

    features = torch.randn(n_nodes, 16)
    features[fraud_nodes] += 2   # fraud nodes have different feature distribution

    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[fraud_nodes] = 1

    return Graph(n_nodes, edge_index, features, labels)


# ── Training ──────────────────────────────────────────────────────────────

def train_node_classifier(graph, model_name="GCN", n_epochs=200, hidden=64):
    graph = graph.to(DEVICE)
    n_feat    = graph.node_features.shape[1]
    n_classes = graph.labels.max().item() + 1

    A_norm = graph.get_adjacency().to(DEVICE)

    if model_name == "GCN":
        model = GCN(n_feat, hidden, n_classes).to(DEVICE)
    else:
        model = nn.Sequential(
            SAGELayer(n_feat, hidden),
        )
        model = GCN(n_feat, hidden, n_classes).to(DEVICE)  # fallback

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 60/20/20 split
    n = graph.n_nodes
    idx = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(0.6*n)]]  = True
    val_mask[idx[int(0.6*n):int(0.8*n)]] = True
    test_mask[idx[int(0.8*n):]]  = True

    train_mask = train_mask.to(DEVICE)
    test_mask  = test_mask.to(DEVICE)

    train_losses, train_accs = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(graph.node_features, A_norm)
        loss = F.nll_loss(out[train_mask], graph.labels[train_mask])
        loss.backward()
        optimizer.step()

        acc = (out[train_mask].argmax(1) == graph.labels[train_mask]).float().mean()
        train_losses.append(loss.item())
        train_accs.append(acc.item())

    model.eval()
    with torch.no_grad():
        out = model(graph.node_features, A_norm)
        test_acc = (out[test_mask].argmax(1) == graph.labels[test_mask]).float().mean()

    return model, train_losses, train_accs, test_acc.item()


def main():
    print("=== Graph Neural Networks (GNNs) ===")
    print(f"Device: {DEVICE}\n")

    print("── Core Concept: Message Passing ──\n")
    print("""
  Traditional ML:  features of one sample are independent
  GNN:             features flow between connected nodes

  Message passing (one iteration):
    For each node v:
      1. Collect messages from neighbors: m_v = AGG({h_u : u ∈ N(v)})
      2. Update own state:                h_v_new = UPDATE(h_v, m_v)

  After K iterations, each node has seen its K-hop neighborhood.
  K=2 on a social network: friends of friends.

  Applications:
    Social networks:  classify users (bot/human, influential/passive)
    Molecules:        predict toxicity, solubility (each atom = node)
    Fraud detection:  transactions sharing cards/merchants form graphs
    Citation graphs:  predict paper topic from citations
    Knowledge graphs: predict missing facts (Google Knowledge Graph)
    """)

    # ── Task 1: Social network community detection ─────────────────────
    print("── Task 1: Community Detection (Social Network) ──\n")
    g1 = create_karate_like_graph()
    print(f"Nodes: {g1.n_nodes}, Edges: {g1.edge_index.shape[1]}")
    print(f"Community A: {(g1.labels==0).sum()}, Community B: {(g1.labels==1).sum()}\n")

    _, losses1, accs1, test_acc1 = train_node_classifier(g1, n_epochs=200, hidden=32)
    print(f"Community detection test accuracy: {test_acc1:.1%}\n")

    # ── Task 2: Fraud detection graph ─────────────────────────────────
    print("── Task 2: Fraud Detection (Transaction Graph) ──\n")
    g2 = create_fraud_graph()
    print(f"Nodes: {g2.n_nodes}, Edges: {g2.edge_index.shape[1]}")
    print(f"Normal: {(g2.labels==0).sum()}, Fraud: {(g2.labels==1).sum()}\n")

    _, losses2, accs2, test_acc2 = train_node_classifier(g2, n_epochs=200, hidden=64)
    print(f"Fraud detection test accuracy: {test_acc2:.1%}\n")

    # ── Visualize training curves ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (losses, accs, acc, title) in enumerate([
        (losses1, accs1, test_acc1, "Community Detection"),
        (losses2, accs2, test_acc2, "Fraud Detection"),
    ]):
        axes[row, 0].plot(losses, color="red")
        axes[row, 0].set_title(f"{title} — Training Loss")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("NLL Loss")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot([a*100 for a in accs], color="blue")
        axes[row, 1].axhline(y=acc*100, color="green", linestyle="--",
                             label=f"Test acc: {acc:.1%}")
        axes[row, 1].set_title(f"{title} — Training Accuracy")
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].set_ylabel("Accuracy (%)")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)

    plt.suptitle("GCN Node Classification", fontsize=13)
    plt.tight_layout()
    plt.savefig("25_gnn_training.png", dpi=100)
    print("Saved training curves to 25_gnn_training.png")

    # ── Adjacency matrix visualization ────────────────────────────────
    A1 = g1.get_adjacency().numpy()
    plt.figure(figsize=(8, 7))
    plt.imshow(A1, cmap="Blues", aspect="auto")
    plt.colorbar()
    plt.title("Normalized Adjacency Matrix\n"
              "(block structure = two communities)")
    plt.xlabel("Node index")
    plt.ylabel("Node index")
    plt.tight_layout()
    plt.savefig("25_gnn_adjacency.png", dpi=100)
    print("Saved adjacency matrix to 25_gnn_adjacency.png")

    print("\n── Next Steps with GNNs ──")
    print("  PyTorch Geometric (PyG): pip install torch-geometric")
    print("  - GATConv  (Graph Attention Network)")
    print("  - GINConv  (Graph Isomorphism Network)")
    print("  - MessagePassing base class for custom layers")
    print("  - Built-in datasets: Cora, CiteSeer, PubMed, ZINC, OGBN")
    print("\n  DGL (Deep Graph Library): alternative to PyG")

    print("\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Graph representation — nodes, edges, adjacency matrix")
    print("  ✓ Normalized adjacency — D^{-1/2}(A+I)D^{-1/2}")
    print("  ✓ GCN layer — neighbor aggregation + linear transform")
    print("  ✓ Message passing — K hops = K-layer GCN")
    print("  ✓ Node classification — predict labels for each node")
    print("  ✓ Semi-supervised learning — few labeled nodes, many unlabeled")


if __name__ == "__main__":
    main()
