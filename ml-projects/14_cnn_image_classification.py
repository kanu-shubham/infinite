"""
Project 14: CNNs for Image Classification
===========================================
Build a Convolutional Neural Network from scratch using PyTorch
to classify images from CIFAR-10 (10 classes: plane, car, bird, cat, ...).

What you'll learn:
- Why CNNs beat fully-connected nets on images
- Conv2d, MaxPool2d, BatchNorm2d — what each layer does
- How to calculate output sizes through a CNN
- Data augmentation to prevent overfitting
- Visualizing filters and feature maps
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# ── Model definitions ─────────────────────────────────────────────────────

class FullyConnectedNet(nn.Module):
    """Baseline: flatten image, feed to FC layers. Ignores spatial structure."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """
    3-block CNN. Each block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool

    Key concepts:
    - Conv2d(in_ch, out_ch, kernel): learns spatial filters
    - BatchNorm2d: normalizes after each conv (stable training)
    - MaxPool2d: downsamples spatially (reduces computation, adds invariance)
    - Global Average Pooling: replaces huge FC layer at the end
    """
    def __init__(self):
        super().__init__()

        # Block 1: 3 → 32 channels, 32×32 → 16×16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (B,32,16,16)
            nn.Dropout2d(0.1),
        )

        # Block 2: 32 → 64 channels, 16×16 → 8×8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (B,64,8,8)
            nn.Dropout2d(0.2),
        )

        # Block 3: 64 → 128 channels, 8×8 → 4×4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B,128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (B,128,4,4)
            nn.Dropout2d(0.3),
        )

        # Classifier: Global Average Pooling → Linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B,128,1,1) — pools each channel to 1 value
            nn.Flatten(),              # (B,128)
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(batch_size=128, data_dir="./data"):
    # Training: augment to reduce overfitting
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                             (0.2023, 0.1994, 0.2010)),  # CIFAR-10 std
    ])

    # Test: no augmentation, just normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def main():
    print("=== CNN Image Classification (CIFAR-10) ===")
    print(f"Device: {DEVICE}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("Downloading CIFAR-10...")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Visualize sample images
    xb, yb = next(iter(train_loader))
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    for i, ax in enumerate(axes.flat):
        img = (xb[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(CLASSES[yb[i]], fontsize=8)
        ax.axis("off")
    plt.suptitle("CIFAR-10 Samples (with augmentation)", fontsize=12)
    plt.tight_layout()
    plt.savefig("14_cifar10_samples.png", dpi=100)
    print("Saved samples to 14_cifar10_samples.png")

    # ── 2. Compare FC vs CNN ─────────────────────────────────────────────
    fc_model  = FullyConnectedNet().to(DEVICE)
    cnn_model = SimpleCNN().to(DEVICE)

    print(f"\nFC  Net parameters:  {count_params(fc_model):,}")
    print(f"CNN Net parameters:  {count_params(cnn_model):,}")
    print(cnn_model)

    # ── 3. Train the CNN ─────────────────────────────────────────────────
    print("\n=== Training SimpleCNN ===\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Cosine annealing: smoothly decay LR each epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    n_epochs = 30
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(cnn_model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc = evaluate(cnn_model, test_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: {tr_acc:.1%} ({tr_loss:.3f}) | "
                  f"Test: {te_acc:.1%} ({te_loss:.3f}) | "
                  f"LR: {lr:.5f}")

    # ── 4. Per-class accuracy ────────────────────────────────────────────
    print("\n=== Per-Class Accuracy ===\n")
    cnn_model.eval()
    class_correct = [0] * 10
    class_total   = [0] * 10

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = cnn_model(xb).argmax(1)
            for pred, label in zip(preds, yb):
                class_correct[label] += (pred == label).item()
                class_total[label]   += 1

    for i, cls in enumerate(CLASSES):
        acc = class_correct[i] / class_total[i]
        bar = "█" * int(acc * 20)
        print(f"  {cls:8s}: {acc:.1%}  {bar}")

    # ── 5. Training curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, n_epochs + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"],  label="Test")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["test_acc"]],  label="Test")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SimpleCNN Training on CIFAR-10", fontsize=13)
    plt.tight_layout()
    plt.savefig("14_cnn_training_curves.png", dpi=100)
    print("\nSaved training curves to 14_cnn_training_curves.png")

    # ── 6. Visualize learned conv filters ────────────────────────────────
    filters = cnn_model.block1[0].weight.data.cpu()  # first conv layer (32, 3, 3, 3)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            f = filters[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f.permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle("Learned Conv Filters (First Layer)", fontsize=12)
    plt.tight_layout()
    plt.savefig("14_cnn_filters.png", dpi=100)
    print("Saved filters to 14_cnn_filters.png")

    # ── 7. Save model ────────────────────────────────────────────────────
    torch.save(cnn_model.state_dict(), "14_cifar10_cnn.pt")

    best_acc = max(history["test_acc"])
    print(f"\n=== Final Results ===")
    print(f"Best test accuracy: {best_acc:.1%}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.1%}")
    print(f"\nKey concepts mastered:")
    print("  ✓ Conv2d — learns spatial patterns (edges, textures, shapes)")
    print("  ✓ BatchNorm — stabilizes training")
    print("  ✓ MaxPool — downsamples, adds translation invariance")
    print("  ✓ Data augmentation — flip, crop, color jitter")
    print("  ✓ AdaptiveAvgPool — replaces large FC layers")
    print("  ✓ Cosine LR scheduler — smooth learning rate decay")


if __name__ == "__main__":
    main()
