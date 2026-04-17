"""
Project 23: Neural Network from Scratch (NumPy Only)
======================================================
Build a complete neural network — forward pass, backpropagation,
and gradient descent — using ONLY NumPy. No PyTorch, no Keras.

This is the most important exercise for understanding deep learning.
Once you build it by hand, PyTorch will make complete sense.

What you'll learn:
- Forward pass: matrix multiplication + activations
- Loss function: cross-entropy, how it measures error
- Backpropagation: chain rule applied layer by layer
- Gradient descent: how weights update
- Numerical gradient checking: verifying your backprop is correct
- The exact math that PyTorch's autograd does automatically
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Activation functions ──────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_backward(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_backward(z):
    return (z > 0).astype(float)

def softmax(z):
    # Numerically stable: subtract max before exp
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ── Loss functions ────────────────────────────────────────────────────────

def cross_entropy_loss(y_pred_proba, y_true_onehot):
    """Cross-entropy: measures how wrong the predicted probabilities are."""
    n = y_true_onehot.shape[0]
    # Clip to avoid log(0)
    log_probs = np.log(y_pred_proba.clip(1e-12, 1 - 1e-12))
    return -np.sum(y_true_onehot * log_probs) / n

def to_onehot(y, n_classes):
    n = len(y)
    oh = np.zeros((n, n_classes))
    oh[np.arange(n), y] = 1
    return oh


# ── Neural Network class ──────────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected neural network built with pure NumPy.

    Architecture: Input → [Hidden layers with ReLU] → Output (Softmax)

    Each layer stores:
      W: weight matrix  (fan_in, fan_out)
      b: bias vector    (1, fan_out)
    """

    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        """
        layer_sizes: list like [input_dim, 64, 32, n_classes]
        """
        np.random.seed(seed)
        self.layer_sizes   = layer_sizes
        self.learning_rate = learning_rate
        self.n_layers      = len(layer_sizes) - 1
        self.params        = {}
        self.cache         = {}  # stores intermediate values for backprop

        # He initialization: scale weights by sqrt(2/fan_in) for ReLU
        for l in range(1, len(layer_sizes)):
            fan_in  = layer_sizes[l - 1]
            fan_out = layer_sizes[l]
            self.params[f"W{l}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{l}"] = np.zeros((1, fan_out))

    def forward(self, X):
        """
        Forward pass: compute predictions.
        Store intermediate values needed for backprop.

        For layer l:
          Z = A_prev @ W + b      (linear combination)
          A = ReLU(Z)             (activation, except last layer)
        Last layer: A = softmax(Z) (output probabilities)
        """
        self.cache["A0"] = X

        for l in range(1, self.n_layers + 1):
            A_prev = self.cache[f"A{l-1}"]
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            Z = A_prev @ W + b           # linear step
            self.cache[f"Z{l}"] = Z

            if l == self.n_layers:
                A = softmax(Z)           # output layer
            else:
                A = relu(Z)              # hidden layers
            self.cache[f"A{l}"] = A

        return self.cache[f"A{self.n_layers}"]

    def backward(self, y_onehot):
        """
        Backward pass: compute gradients via the chain rule.

        Chain rule for layer l:
          dL/dW = A_prev.T @ dZ
          dL/db = mean(dZ, axis=0)
          dL/dA_prev = dZ @ W.T   (propagated to next layer)

        Output layer gradient (softmax + cross-entropy combined):
          dZ_last = y_pred - y_true   (elegant closed form!)
        """
        n = y_onehot.shape[0]
        self.grads = {}

        # Gradient from loss through softmax (closed-form, very clean)
        y_pred = self.cache[f"A{self.n_layers}"]
        dA = y_pred - y_onehot    # dL/dA_last (before softmax, after combining)

        for l in reversed(range(1, self.n_layers + 1)):
            A_prev = self.cache[f"A{l-1}"]
            W      = self.params[f"W{l}"]
            Z      = self.cache[f"Z{l}"]

            if l == self.n_layers:
                dZ = dA                  # softmax+CE gradient already computed
            else:
                dZ = dA * relu_backward(Z)   # chain rule through ReLU

            # Gradients for this layer's parameters
            self.grads[f"W{l}"] = A_prev.T @ dZ / n
            self.grads[f"b{l}"] = dZ.mean(axis=0, keepdims=True)

            # Gradient to pass to previous layer
            dA = dZ @ W.T

    def update(self):
        """Gradient descent: W = W - lr * dL/dW"""
        for l in range(1, self.n_layers + 1):
            self.params[f"W{l}"] -= self.learning_rate * self.grads[f"W{l}"]
            self.params[f"b{l}"] -= self.learning_rate * self.grads[f"b{l}"]

    def predict(self, X):
        proba = self.forward(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X):
        return self.forward(X)

    def compute_loss(self, X, y_onehot):
        y_pred = self.forward(X)
        return cross_entropy_loss(y_pred, y_onehot)


# ── Gradient checking ─────────────────────────────────────────────────────

def gradient_check(nn_model, X, y_oh, epsilon=1e-5):
    """
    Verify backprop by comparing analytical gradients to numerical gradients.
    Numerical gradient: (L(W+ε) - L(W-ε)) / (2ε)
    If they match (relative error < 1e-5), backprop is correct.
    """
    print("\n── Gradient Check (verifying backprop correctness) ──\n")

    # Compute analytical gradients
    nn_model.forward(X)
    nn_model.backward(y_oh)

    # Check a sample of weights from layer 1
    W1 = nn_model.params["W1"]
    dW1_analytical = nn_model.grads["W1"]

    indices = [(0, 0), (0, 1), (1, 0), (2, 1), (3, 0)]
    max_rel_error = 0.0

    print(f"  {'Index':<12} {'Analytical':>15} {'Numerical':>15} {'Rel Error':>12}")
    print("  " + "-" * 56)

    for i, j in indices:
        # + epsilon
        W1[i, j] += epsilon
        loss_plus = nn_model.compute_loss(X, y_oh)
        W1[i, j] -= epsilon

        # - epsilon
        W1[i, j] -= epsilon
        loss_minus = nn_model.compute_loss(X, y_oh)
        W1[i, j] += epsilon

        numerical  = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = dW1_analytical[i, j]
        denom = max(abs(numerical) + abs(analytical), 1e-12)
        rel_error = abs(numerical - analytical) / denom
        max_rel_error = max(max_rel_error, rel_error)

        status = "✓" if rel_error < 1e-5 else "✗"
        print(f"  W[{i},{j}]{'':<5} {analytical:>15.8f} {numerical:>15.8f} {rel_error:>11.2e} {status}")

    print(f"\n  Max relative error: {max_rel_error:.2e}")
    if max_rel_error < 1e-5:
        print("  Backprop is CORRECT ✓")
    else:
        print("  Backprop has a bug ✗")


# ── Training loop ─────────────────────────────────────────────────────────

def train(model, X_train, y_train_oh, X_test, y_test_oh,
          n_epochs=100, batch_size=64, verbose=True):
    """Mini-batch stochastic gradient descent training loop."""
    n = len(X_train)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, n_epochs + 1):
        # Shuffle training data each epoch
        indices = np.random.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]

        # Mini-batch gradient descent
        for start in range(0, n, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            model.forward(X_batch)     # 1. forward pass
            model.backward(y_batch)    # 2. compute gradients
            model.update()             # 3. update weights

        # Record metrics
        tr_loss = model.compute_loss(X_train, y_train_oh)
        te_loss = model.compute_loss(X_test,  y_test_oh)
        tr_acc  = (model.predict(X_train) == y_train_oh.argmax(1)).mean()
        te_acc  = (model.predict(X_test)  == y_test_oh.argmax(1)).mean()

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"Loss: {tr_loss:.4f} | "
                  f"Train: {tr_acc:.1%} | Test: {te_acc:.1%}")

    return history


def main():
    print("=== Neural Network from Scratch (NumPy Only) ===\n")

    # ── Part 1: Explain backpropagation ───────────────────────────────────
    print("── How Backpropagation Works ──\n")
    print("""
  Goal: adjust weights to minimize the loss.

  Forward pass (left to right):
    X → Z1 = X@W1+b1 → A1=ReLU(Z1) → Z2=A1@W2+b2 → A2=softmax(Z2) → Loss

  Backward pass (right to left, chain rule):
    dL/dW2 = A1.T @ (A2 - y)          ← gradient of output layer
    dL/dW1 = X.T  @ ((A2-y) @ W2.T * relu'(Z1))  ← propagated back

  Chain rule: dL/dW1 = dL/dA2 · dA2/dZ2 · dZ2/dA1 · dA1/dZ1 · dZ1/dW1

  PyTorch's autograd does this automatically.
  Building it by hand makes you understand exactly what happens.
    """)

    # ── Part 2: Train on Moons (2 classes) ───────────────────────────────
    print("── Part 2: Moons Dataset ──\n")

    X_moons, y_moons = make_moons(n_samples=1000, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X_moons = scaler.fit_transform(X_moons)

    X_tr, X_te, y_tr, y_te = train_test_split(X_moons, y_moons,
                                               test_size=0.2, random_state=42)
    y_tr_oh = to_onehot(y_tr, n_classes=2)
    y_te_oh = to_onehot(y_te, n_classes=2)

    # Build network: 2 → 16 → 16 → 2
    model_moons = NeuralNetwork([2, 16, 16, 2], learning_rate=0.05)
    print("Architecture: 2 → 16 → 16 → 2")
    total_params = sum(
        model_moons.params[k].size
        for k in model_moons.params
    )
    print(f"Total parameters: {total_params}\n")

    # Gradient check before training
    gradient_check(model_moons, X_tr[:20], y_tr_oh[:20])

    print("\nTraining on Moons...")
    history = train(model_moons, X_tr, y_tr_oh, X_te, y_te_oh,
                    n_epochs=200, batch_size=32)

    # Visualize decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    Z = model_moons.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
    axes[0].scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="RdBu",
                    edgecolors="k", linewidth=0.5, s=30)
    axes[0].set_title(f"Learned Decision Boundary\nTest acc: {history['test_acc'][-1]:.1%}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train loss")
    axes[1].plot(history["test_loss"],  label="Test loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Neural Network from Scratch — Moons", fontsize=13)
    plt.tight_layout()
    plt.savefig("23_nn_scratch_moons.png", dpi=100)
    print("\nSaved moons results to 23_nn_scratch_moons.png")

    # ── Part 3: Train on Digits (10 classes) ─────────────────────────────
    print("\n── Part 3: Digits Dataset (10 classes) ──\n")

    digits = load_digits()
    X_d = StandardScaler().fit_transform(digits.data)
    y_d = digits.target

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d,
                                                       test_size=0.2, random_state=42)
    y_tr_oh_d = to_onehot(y_tr_d, n_classes=10)
    y_te_oh_d = to_onehot(y_te_d, n_classes=10)

    # Deeper network for 10 classes: 64 → 128 → 64 → 10
    model_digits = NeuralNetwork([64, 128, 64, 10], learning_rate=0.01)
    print("Architecture: 64 → 128 → 64 → 10")
    total_params = sum(model_digits.params[k].size for k in model_digits.params)
    print(f"Total parameters: {total_params}\n")

    history_d = train(model_digits, X_tr_d, y_tr_oh_d, X_te_d, y_te_oh_d,
                      n_epochs=300, batch_size=64)

    final_acc = history_d["test_acc"][-1]
    print(f"\nFinal test accuracy: {final_acc:.1%}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_d["train_loss"], label="Train")
    axes[0].plot(history_d["test_loss"],  label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([a * 100 for a in history_d["train_acc"]], label="Train")
    axes[1].plot([a * 100 for a in history_d["test_acc"]],  label="Test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Accuracy Curve (best: {final_acc:.1%})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Neural Network from Scratch — Digits (10 classes)", fontsize=13)
    plt.tight_layout()
    plt.savefig("23_nn_scratch_digits.png", dpi=100)
    print("Saved digits results to 23_nn_scratch_digits.png")

    # ── Part 4: Compare with different learning rates ─────────────────────
    print("\n── Part 4: Effect of Learning Rate ──\n")

    lrs = [0.001, 0.01, 0.1, 0.5]
    fig, ax = plt.subplots(figsize=(10, 5))

    for lr in lrs:
        m = NeuralNetwork([2, 16, 16, 2], learning_rate=lr)
        h = train(m, X_tr, y_tr_oh, X_te, y_te_oh,
                  n_epochs=100, batch_size=32, verbose=False)
        ax.plot(h["test_loss"], label=f"lr={lr}")
        print(f"  lr={lr} | Final test acc: {h['test_acc'][-1]:.1%}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title("Effect of Learning Rate on Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("23_nn_learning_rate.png", dpi=100)
    print("\nSaved learning rate comparison to 23_nn_learning_rate.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Forward pass — Z = X@W+b, A = ReLU(Z)")
    print("  ✓ Softmax — converts logits to probabilities")
    print("  ✓ Cross-entropy loss — penalizes wrong predictions")
    print("  ✓ Backpropagation — chain rule layer by layer")
    print("  ✓ dZ_last = y_pred - y_true — elegant softmax+CE gradient")
    print("  ✓ He initialization — proper weight scaling for ReLU")
    print("  ✓ Mini-batch SGD — faster than full-batch gradient descent")
    print("  ✓ Gradient checking — verify backprop numerically")
    print("\nNow when PyTorch calls loss.backward(), you know exactly what it does.")


if __name__ == "__main__":
    main()
