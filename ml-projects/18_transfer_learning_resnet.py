"""
Project 18: Transfer Learning with ResNet
==========================================
Fine-tune a ResNet-18 pretrained on ImageNet to classify a new dataset.
Learn the two strategies: feature extraction vs full fine-tuning.

What you'll learn:
- Why transfer learning works (features generalize across tasks)
- ResNet architecture: residual connections solve vanishing gradients
- Feature extraction: freeze backbone, train only the head
- Fine-tuning: unfreeze all layers with a small learning rate
- Comparing training from scratch vs transfer learning
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use CIFAR-10 but treat it as a 2-class problem (animals vs vehicles)
# to demonstrate fine-tuning on a "different" distribution
ANIMAL_CLASSES   = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse
VEHICLE_CLASSES  = {0, 1, 8, 9}         # plane, car, ship, truck
ALL_CLASSES      = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def explain_resnet():
    print("── ResNet: Residual Connections ──\n")
    print("""
  Normal block:     x → [Conv → BN → ReLU → Conv → BN] → ReLU → out
  Residual block:   x → [Conv → BN → ReLU → Conv → BN] → (+x) → ReLU → out
                                                            ↑
                                                      Skip connection

  Why it works:
  - Deep networks (50+ layers) suffer from vanishing gradients
  - Skip connection creates a "gradient highway" back to earlier layers
  - In the worst case, layers learn identity (do nothing) — safe default
  - Makes 100+ layer networks trainable

  Key variants:
  ResNet-18/34:  Basic blocks (2 conv per block)
  ResNet-50+:    Bottleneck blocks (1×1 → 3×3 → 1×1 convs)

  Pretrained on ImageNet: 1.28M images, 1000 classes, 1000 GPU-hours
  → You get those learned features for free via transfer learning
    """)


def get_cifar_loaders(batch_size=64):
    """CIFAR-10 but upsized to 224x224 for ResNet compatibility."""
    # ResNet expects 224×224 ImageNet-normalized inputs
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    # Subsample for speed (use 5000 train, 1000 test)
    torch.manual_seed(42)
    train_idx = torch.randperm(len(train_ds))[:5000]
    test_idx  = torch.randperm(len(test_ds))[:1000]

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_resnet(strategy="feature_extraction", n_classes=10):
    """
    Build ResNet-18 with one of two transfer learning strategies.

    strategy = "feature_extraction":
        Freeze ALL pretrained layers → only train the final FC layer
        Fast, good when target domain is similar to ImageNet

    strategy = "fine_tuning":
        Unfreeze ALL layers → train everything (with small LR)
        Slower, better when you have enough data or different domain

    strategy = "scratch":
        Random initialization → train from zero (baseline comparison)
    """
    if strategy == "scratch":
        model = torchvision.models.resnet18(weights=None)
    else:
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )

    if strategy == "feature_extraction":
        # Freeze all pretrained layers
        for param in model.parameters():
            param.requires_grad = False
        # Only the new head is trainable
        print("  Frozen layers: all (except new head)")

    # Replace the final FC layer (ImageNet has 1000 classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, n_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    return model.to(DEVICE)


def train_model(model, train_loader, test_loader, n_epochs, lr, name):
    """Training loop that returns history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_acc": [], "test_acc": [], "time": []}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
        train_acc = correct / total

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        test_acc = correct / total

        scheduler.step()
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["time"].append(time.time() - t0)

        if epoch % 3 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{n_epochs} | Train: {train_acc:.1%} | Test: {test_acc:.1%}")

    return history


def visualize_feature_maps(model, loader):
    """Show what ResNet's layers activate on."""
    print("\n── Feature Map Visualization ──\n")
    model.eval()
    xb, _ = next(iter(loader))

    # Hook to capture intermediate activations
    activations = {}
    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    model.layer1.register_forward_hook(make_hook("layer1"))
    model.layer4.register_forward_hook(make_hook("layer4"))

    with torch.no_grad():
        model(xb[:1].to(DEVICE))

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))

    # Layer 1 feature maps (shallow — edges/textures)
    feat1 = activations["layer1"][0]
    for i, ax in enumerate(axes[0]):
        ax.imshow(feat1[i].numpy(), cmap="viridis")
        ax.set_title(f"L1[{i}]", fontsize=7)
        ax.axis("off")

    # Layer 4 feature maps (deep — high-level semantics)
    feat4 = activations["layer4"][0]
    for i, ax in enumerate(axes[1]):
        ax.imshow(feat4[i].numpy(), cmap="viridis")
        ax.set_title(f"L4[{i}]", fontsize=7)
        ax.axis("off")

    axes[0, 0].set_ylabel("Layer 1\n(shallow)", fontsize=9)
    axes[1, 0].set_ylabel("Layer 4\n(deep)", fontsize=9)

    plt.suptitle("ResNet Feature Maps: Shallow vs Deep Layers", fontsize=12)
    plt.tight_layout()
    plt.savefig("18_resnet_feature_maps.png", dpi=100)
    print("Saved feature maps to 18_resnet_feature_maps.png")


def main():
    print("=== Transfer Learning with ResNet ===")
    print(f"Device: {DEVICE}\n")

    explain_resnet()

    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}\n")

    N_EPOCHS = 10
    results  = {}

    strategies = [
        ("scratch",            1e-3, "From Scratch (no pretrained)"),
        ("feature_extraction", 1e-3, "Feature Extraction (frozen backbone)"),
        ("fine_tuning",        1e-4, "Fine-Tuning (all layers, small LR)"),
    ]

    for strategy, lr, label in strategies:
        print(f"\n── {label} ──")
        model = build_resnet(strategy=strategy, n_classes=10)
        history = train_model(model, train_loader, test_loader, N_EPOCHS, lr, label)
        results[label] = history

        best_acc = max(history["test_acc"])
        print(f"  Best test accuracy: {best_acc:.1%}")
        torch.save(model.state_dict(), f"18_resnet_{strategy}.pt")

    # ── Comparison plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for (label, hist), color in zip(results.items(), colors):
        epochs = range(1, N_EPOCHS + 1)
        axes[0].plot(epochs, [a * 100 for a in hist["test_acc"]], label=label, color=color, linewidth=2)
        axes[1].plot(hist["time"], [a * 100 for a in hist["test_acc"]], label=label, color=color, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Test Accuracy per Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Wall-clock Time (s)")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Accuracy vs. Training Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Transfer Learning Strategies on CIFAR-10", fontsize=13)
    plt.tight_layout()
    plt.savefig("18_transfer_learning_comparison.png", dpi=100)
    print("\nSaved comparison to 18_transfer_learning_comparison.png")

    # Feature maps
    best_model = build_resnet(strategy="fine_tuning", n_classes=10)
    best_model.load_state_dict(torch.load("18_resnet_fine_tuning.pt", map_location=DEVICE))
    visualize_feature_maps(best_model, test_loader)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n=== Final Comparison ===\n")
    print(f"{'Strategy':<40} {'Best Acc':>10} {'Time(s)':>10}")
    print("-" * 62)
    for label, hist in results.items():
        best_acc = max(hist["test_acc"])
        total_t  = hist["time"][-1]
        print(f"{label:<40} {best_acc:>9.1%} {total_t:>9.1f}s")

    print(f"\nKey takeaways:")
    print("  - Feature extraction: fast, good for similar domains")
    print("  - Fine-tuning: best accuracy, needs careful LR tuning")
    print("  - From scratch: worst accuracy with limited data")
    print("  - Rule: always try feature extraction first, then fine-tune")


if __name__ == "__main__":
    main()
