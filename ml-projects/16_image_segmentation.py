"""
Project 16: Image Segmentation
================================
Classify every pixel in an image. Two types:
- Semantic segmentation: label each pixel (sky, road, car...)
- Instance segmentation: separate individual objects

What you'll learn:
- Difference: Classification → Detection → Segmentation
- U-Net architecture (encoder-decoder with skip connections)
- Pretrained DeepLabV3 from torchvision
- How skip connections preserve spatial detail
- Evaluating with IoU per class (mIoU)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOC 2012 segmentation classes
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_COLORS = plt.cm.tab20(np.linspace(0, 1, len(VOC_CLASSES)))


# ── Part 1: U-Net from scratch ────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two Conv→BN→ReLU blocks — the basic unit of U-Net."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net: Encoder–Decoder with Skip Connections.

    Architecture:
      Encoder (contracting path): doubles channels, halves spatial size
      Bottleneck: deepest feature representation
      Decoder (expanding path): halves channels, doubles spatial size
      Skip connections: paste encoder features onto decoder (preserves detail)

    Input:  (B, in_ch, H, W)
    Output: (B, n_classes, H, W)  — one score map per class per pixel
    """
    def __init__(self, in_ch=3, n_classes=21):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (upsample + concatenate skip + double conv)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 because of skip concat

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output: 1x1 conv maps to n_classes
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder — save outputs for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder — upsample then concatenate with encoder output
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


def visualize_unet_architecture():
    """Show how data flows through U-Net by tracing shapes."""
    print("\n── U-Net Shape Trace (256×256 input) ──\n")

    model = UNet(in_ch=3, n_classes=21).to(DEVICE)
    x = torch.randn(1, 3, 256, 256).to(DEVICE)

    # Manual forward pass with shape printing
    pool = model.pool
    e1 = model.enc1(x);         print(f"  Encoder 1:     {x.shape} → {e1.shape}")
    e2 = model.enc2(pool(e1));  print(f"  Encoder 2:     {pool(e1).shape} → {e2.shape}")
    e3 = model.enc3(pool(e2));  print(f"  Encoder 3:     {pool(e2).shape} → {e3.shape}")
    e4 = model.enc4(pool(e3));  print(f"  Encoder 4:     {pool(e3).shape} → {e4.shape}")
    b  = model.bottleneck(pool(e4)); print(f"  Bottleneck:    {pool(e4).shape} → {b.shape}")
    up4 = model.up4(b); d4 = model.dec4(torch.cat([up4, e4], 1))
    print(f"  Decoder 4:     up:{up4.shape} + skip:{e4.shape} → {d4.shape}")
    up3 = model.up3(d4); d3 = model.dec3(torch.cat([up3, e3], 1))
    print(f"  Decoder 3:     up:{up3.shape} + skip:{e3.shape} → {d3.shape}")
    up2 = model.up2(d3); d2 = model.dec2(torch.cat([up2, e2], 1))
    print(f"  Decoder 2:     up:{up2.shape} + skip:{e2.shape} → {d2.shape}")
    up1 = model.up1(d2); d1 = model.dec1(torch.cat([up1, e1], 1))
    print(f"  Decoder 1:     up:{up1.shape} + skip:{e1.shape} → {d1.shape}")
    out = model.out_conv(d1)
    print(f"  Output:        {out.shape}  (1 score map per class per pixel)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nU-Net parameters: {n_params:,}")
    return model


# ── Part 2: Synthetic segmentation training demo ──────────────────────────

def create_synthetic_seg_dataset(n=200, size=64, n_classes=4):
    """
    Generate simple (image, mask) pairs:
    - Class 0: background (gray)
    - Class 1: red rectangle
    - Class 2: green circle
    - Class 3: blue triangle
    """
    images, masks = [], []

    for _ in range(n):
        img = np.ones((3, size, size), dtype=np.float32) * 0.5
        mask = np.zeros((size, size), dtype=np.int64)

        # Random rectangle (class 1)
        if np.random.rand() > 0.3:
            x1, y1 = np.random.randint(0, size//2, 2)
            x2, y2 = x1 + np.random.randint(10, 25), y1 + np.random.randint(10, 25)
            x2, y2 = min(x2, size-1), min(y2, size-1)
            img[0, y1:y2, x1:x2] = 0.9
            img[1, y1:y2, x1:x2] = 0.2
            img[2, y1:y2, x1:x2] = 0.2
            mask[y1:y2, x1:x2] = 1

        # Random circle (class 2)
        if np.random.rand() > 0.3:
            cx, cy = np.random.randint(15, size-15, 2)
            r = np.random.randint(8, 15)
            Y, X = np.ogrid[:size, :size]
            circle = (X - cx)**2 + (Y - cy)**2 <= r**2
            img[0, circle] = 0.2
            img[1, circle] = 0.8
            img[2, circle] = 0.2
            mask[circle] = 2

        # Random square blob (class 3)
        if np.random.rand() > 0.3:
            x1, y1 = np.random.randint(size//2, size-20, 2)
            side = np.random.randint(8, 18)
            x2 = min(x1 + side, size-1)
            y2 = min(y1 + side, size-1)
            img[0, y1:y2, x1:x2] = 0.2
            img[1, y1:y2, x1:x2] = 0.2
            img[2, y1:y2, x1:x2] = 0.9
            mask[y1:y2, x1:x2] = 3

        images.append(torch.tensor(img))
        masks.append(torch.tensor(mask))

    return torch.stack(images), torch.stack(masks)


def compute_miou(preds, targets, n_classes):
    """Mean Intersection over Union across all classes."""
    ious = []
    for cls in range(n_classes):
        pred_cls   = (preds   == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def train_mini_unet():
    """Train a tiny U-Net on synthetic shapes."""
    print("\n── Training Mini U-Net on Synthetic Shapes ──\n")

    N_CLASSES = 4
    SIZE = 64

    # Dataset
    X, Y = create_synthetic_seg_dataset(n=500, size=SIZE, n_classes=N_CLASSES)

    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split].to(DEVICE), Y[:split].to(DEVICE)
    X_te, Y_te = X[split:].to(DEVICE), Y[split:].to(DEVICE)

    # Tiny U-Net (reduced channels for speed)
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(3, 16)
            self.enc2 = DoubleConv(16, 32)
            self.pool = nn.MaxPool2d(2)
            self.bot  = DoubleConv(32, 64)
            self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = DoubleConv(64, 32)
            self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = DoubleConv(32, 16)
            self.out  = nn.Conv2d(16, N_CLASSES, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b  = self.bot(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b),  e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)

    model = TinyUNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 20
    batch_size = 32
    losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[i:i+batch_size]
            yb = Y_tr[i:i+batch_size]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(1)
                miou = compute_miou(preds.cpu(), Y_te.cpu(), N_CLASSES)
            print(f"  Epoch {epoch:3d}/{n_epochs} | Loss: {epoch_loss:.3f} | Test mIoU: {miou:.3f}")

    # Visualize results
    model.eval()
    with torch.no_grad():
        sample_imgs  = X_te[:4].cpu()
        sample_masks = Y_te[:4].cpu()
        sample_preds = model(X_te[:4]).argmax(1).cpu()

    class_colors = np.array([[0.5, 0.5, 0.5], [0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.9]])

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i in range(4):
        img = sample_imgs[i].permute(1, 2, 0).numpy()
        gt  = class_colors[sample_masks[i].numpy()]
        pr  = class_colors[sample_preds[i].numpy()]

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gt)
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")

        axes[2, i].imshow(pr)
        axes[2, i].set_title("Predicted")
        axes[2, i].axis("off")

    plt.suptitle("U-Net Segmentation Results\n(grey=bg, red=rect, green=circle, blue=square)", fontsize=12)
    plt.tight_layout()
    plt.savefig("16_segmentation_results.png", dpi=100)
    print("\nSaved results to 16_segmentation_results.png")
    return model, losses


# ── Part 3: Pretrained DeepLabV3 ─────────────────────────────────────────

def run_pretrained_segmentation():
    """Run DeepLabV3 (COCO pretrained) for semantic segmentation."""
    print("\n── Pretrained DeepLabV3 (VOC classes) ──\n")

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )
    model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepLabV3 parameters: {n_params:,}")

    # Create a synthetic test image
    img_np = np.random.rand(400, 600, 3).astype(np.float32)
    # Add some structure
    img_np[50:200, 100:300] = [0.8, 0.3, 0.3]   # red region
    img_np[250:380, 350:550] = [0.3, 0.7, 0.3]  # green region
    img = Image.fromarray((img_np * 255).astype(np.uint8))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    pred_mask = output.argmax(0).cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    seg_display = axes[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("DeepLabV3 Segmentation")
    axes[1].axis("off")

    unique_classes = np.unique(pred_mask)
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=plt.cm.tab20(c / 20), label=VOC_CLASSES[c] if c < len(VOC_CLASSES) else f"cls_{c}")
        for c in unique_classes
    ]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("16_deeplabv3_segmentation.png", dpi=100)
    print("Saved DeepLabV3 output to 16_deeplabv3_segmentation.png")


def main():
    print("=== Image Segmentation ===")
    print(f"Device: {DEVICE}\n")

    print("── Task Comparison ──")
    print("  Classification:  one label per image  ('cat')")
    print("  Detection:       boxes per object     ('cat at [100,50,200,150]')")
    print("  Segmentation:    label per pixel      (pixel 57,89 = 'cat')")

    # U-Net architecture
    visualize_unet_architecture()

    # Train on synthetic data
    model, losses = train_mini_unet()

    # Pretrained model
    run_pretrained_segmentation()

    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ U-Net encoder-decoder with skip connections")
    print("  ✓ Skip connections — why they preserve fine detail")
    print("  ✓ Per-pixel cross-entropy loss")
    print("  ✓ mIoU — standard segmentation metric")
    print("  ✓ DeepLabV3 — pretrained semantic segmentation")
    print("\nNext steps:")
    print("  - Mask R-CNN for instance segmentation (torchvision)")
    print("  - SAM (Segment Anything Model) from Meta")


if __name__ == "__main__":
    main()
