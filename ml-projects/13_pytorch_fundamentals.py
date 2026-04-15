"""
Project 13: PyTorch Fundamentals
==================================
Learn PyTorch from zero: tensors, autograd, and building a neural network
entirely from scratch — no sklearn, just PyTorch.

What you'll learn:
- Tensors: PyTorch's core data structure (like NumPy but GPU-ready)
- Autograd: automatic gradient computation (the magic behind backprop)
- nn.Module: how to define a neural network as a class
- Training loop: forward pass → loss → backward → optimizer step
- GPU support: moving tensors to CUDA with .to(device)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ── Helper ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("=== PyTorch Fundamentals ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 1: Tensors — The Building Block")
    # ────────────────────────────────────────────────────────────────────

    # Creating tensors
    scalar = torch.tensor(3.14)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    cube   = torch.zeros(2, 3, 4)                 # shape (2, 3, 4)

    print("scalar:", scalar,        "shape:", scalar.shape)
    print("vector:", vector,        "shape:", vector.shape)
    print("matrix:\n", matrix,      "shape:", matrix.shape)
    print("cube shape:", cube.shape)

    # Tensor operations (all GPU-compatible)
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[5., 6.], [7., 8.]])

    print("\nElement-wise add:\n", a + b)
    print("Matrix multiply:\n", a @ b)         # or torch.matmul(a, b)
    print("Sum:", a.sum(), "  Mean:", a.mean(), "  Max:", a.max())

    # Reshape — critical skill
    x = torch.arange(12, dtype=torch.float32)
    print("\nOriginal:", x.shape)
    print("Reshaped to (3,4):", x.reshape(3, 4).shape)
    print("Unsqueezed:", x.unsqueeze(0).shape)   # add dim → (1, 12)
    print("Squeezed:", x.unsqueeze(0).squeeze().shape)  # remove dim

    # NumPy bridge (zero-copy)
    np_array = np.array([1., 2., 3.])
    torch_tensor = torch.from_numpy(np_array)
    back_to_numpy = torch_tensor.numpy()
    print("\nNumPy → Tensor → NumPy: works seamlessly")

    # Moving to GPU (no-op on CPU machine, but same code works on GPU)
    tensor_on_device = a.to(DEVICE)
    print(f"Tensor device: {tensor_on_device.device}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 2: Autograd — Automatic Differentiation")
    # ────────────────────────────────────────────────────────────────────

    # requires_grad=True tells PyTorch to track operations for backprop
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    # Forward pass: y = w*x + b
    x_val = torch.tensor(3.0)
    y_pred = w * x_val + b
    y_true = torch.tensor(10.0)

    loss = (y_pred - y_true) ** 2   # MSE (single sample)
    print(f"Prediction: {y_pred.item():.4f}")
    print(f"Loss: {loss.item():.4f}")

    # Backward pass: compute gradients
    loss.backward()

    print(f"\nd(loss)/dw = {w.grad.item():.4f}  (expected: 2*(y_pred - y_true)*x)")
    print(f"d(loss)/db = {b.grad.item():.4f}  (expected: 2*(y_pred - y_true))")

    # Manual gradient check
    y_diff = (y_pred - y_true).item()
    expected_dw = 2 * y_diff * x_val.item()
    expected_db = 2 * y_diff
    print(f"\nManual check — dw: {expected_dw:.4f}, db: {expected_db:.4f}  ✓")

    # Gradient accumulates! Must zero before next backward
    print(f"\nGradient before zero_grad: {w.grad.item():.4f}")
    w.grad.zero_()
    b.grad.zero_()
    print(f"Gradient after zero_grad: {w.grad.item():.4f}")

    # torch.no_grad(): disable autograd for inference (saves memory & speed)
    with torch.no_grad():
        y_inference = w * x_val + b
    print(f"\nInference result (no grad): {y_inference.item():.4f}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 3: Building a Neural Network with nn.Module")
    # ────────────────────────────────────────────────────────────────────

    class SimpleNet(nn.Module):
        """A 3-layer fully connected network."""

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()           # always call super().__init__()
            # Define layers as attributes
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            """Define the forward pass. PyTorch calls this automatically."""
            x = self.relu(self.fc1(x))   # layer 1 + activation
            x = self.dropout(x)
            x = self.relu(self.fc2(x))   # layer 2 + activation
            x = self.fc3(x)              # output layer (no activation)
            return x

    model = SimpleNet(input_size=10, hidden_size=64, output_size=1).to(DEVICE)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass through the model
    dummy_input = torch.randn(4, 10).to(DEVICE)   # batch of 4
    dummy_output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 4: Custom Dataset and DataLoader")
    # ────────────────────────────────────────────────────────────────────

    class SineDataset(Dataset):
        """Dataset of (x, sin(x)) pairs — teaches the shape of a sine wave."""

        def __init__(self, n_samples=1000, noise=0.1):
            x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples)
            y = torch.sin(x) + noise * torch.randn(n_samples)
            # Reshape for nn.Linear: (N,) → (N, 1)
            self.X = x.unsqueeze(1)
            self.y = y.unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = SineDataset(n_samples=2000)
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    print(f"Dataset: {len(dataset)} samples")
    print(f"Train:   {len(train_ds)}, Test: {len(test_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Inspect one batch
    xb, yb = next(iter(train_loader))
    print(f"\nBatch x shape: {xb.shape}")
    print(f"Batch y shape: {yb.shape}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 5: The Training Loop")
    # ────────────────────────────────────────────────────────────────────

    class SineNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = SineNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler: reduce LR when plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, test_losses = [], []
    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        # ── Training phase ───────────────────────────────────────────
        model.train()                    # enables dropout, batch norm
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()        # 1. clear gradients
            y_pred = model(xb)           # 2. forward pass
            loss = criterion(y_pred, yb) # 3. compute loss
            loss.backward()              # 4. backward pass
            optimizer.step()             # 5. update weights

            train_loss += loss.item() * len(xb)

        train_loss /= len(train_ds)

        # ── Evaluation phase ─────────────────────────────────────────
        model.eval()                     # disables dropout, batch norm
        test_loss = 0.0

        with torch.no_grad():            # no gradients needed for eval
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                y_pred = model(xb)
                test_loss += criterion(y_pred, yb).item() * len(xb)

        test_loss /= len(test_ds)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{n_epochs} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | LR: {lr:.5f}")

    # ────────────────────────────────────────────────────────────────────
    section("PART 6: Save / Load / Visualize")
    # ────────────────────────────────────────────────────────────────────

    # Save — only the state dict (weights), not the whole model object
    torch.save(model.state_dict(), "13_sine_model.pt")
    print("Model saved to 13_sine_model.pt")

    # Load into a fresh model
    loaded_model = SineNet().to(DEVICE)
    loaded_model.load_state_dict(torch.load("13_sine_model.pt", map_location=DEVICE))
    loaded_model.eval()
    print("Model loaded successfully")

    # Generate predictions
    x_plot = torch.linspace(-2 * np.pi, 2 * np.pi, 500).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        y_plot = loaded_model(x_plot).cpu().numpy()

    x_np = x_plot.cpu().numpy().flatten()
    y_true = np.sin(x_np)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_np, y_true, label="True sin(x)", linewidth=2)
    axes[0].plot(x_np, y_plot, "--", label="Model prediction", linewidth=2)
    axes[0].set_title("Sine Wave: True vs Predicted")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, n_epochs + 1), train_losses, label="Train")
    axes[1].plot(range(1, n_epochs + 1), test_losses, label="Test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Training Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_pytorch_fundamentals.png", dpi=100)
    print("Saved plot to 13_pytorch_fundamentals.png")

    print(f"\n=== Summary ===")
    print(f"Final train loss: {train_losses[-1]:.5f}")
    print(f"Final test loss:  {test_losses[-1]:.5f}")
    print(f"\nKey concepts mastered:")
    print("  ✓ Tensors — shape, dtype, device, operations")
    print("  ✓ Autograd — backward(), .grad, zero_grad()")
    print("  ✓ nn.Module — building nets as classes")
    print("  ✓ Dataset/DataLoader — batching and shuffling")
    print("  ✓ Training loop — the 5-step pattern")
    print("  ✓ Save/Load — state_dict pattern")


if __name__ == "__main__":
    main()
