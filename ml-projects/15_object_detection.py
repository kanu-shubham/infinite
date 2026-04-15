"""
Project 15: Object Detection
==============================
Detect and locate objects in images using pretrained Faster R-CNN from torchvision.
Then explore a lightweight custom detector on synthetic data.

What you'll learn:
- Difference between classification, detection, and segmentation
- Bounding boxes: [x_min, y_min, x_max, y_max] format
- How Faster R-CNN works: backbone + RPN + RoI head
- Using pretrained COCO models for zero-shot detection
- Intersection over Union (IoU) — the key detection metric
- Non-Maximum Suppression (NMS) — removing duplicate boxes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.ops import box_iou, nms
from PIL import Image, ImageDraw
import urllib.request
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO class labels (80 classes)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ── Part 1: Key concepts ──────────────────────────────────────────────────

def explain_iou():
    """Visualize Intersection over Union (IoU)."""
    print("\n── Intersection over Union (IoU) ──\n")

    # Ground truth box
    gt_box = torch.tensor([[100, 100, 300, 250]], dtype=torch.float32)
    # Various predicted boxes
    pred_boxes = torch.tensor([
        [110, 110, 290, 240],  # good overlap
        [200, 150, 400, 350],  # partial overlap
        [400, 300, 550, 450],  # no overlap
    ], dtype=torch.float32)

    ious = box_iou(gt_box, pred_boxes)[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 500)
    ax.invert_yaxis()

    colors = ["green", "orange", "red"]
    labels = ["Good detection", "Partial overlap", "No overlap"]

    # Draw ground truth
    gt = gt_box[0]
    rect = patches.Rectangle(
        (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
        linewidth=3, edgecolor="blue", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(gt[0], gt[1] - 5, "Ground Truth", color="blue", fontsize=10)

    # Draw predicted boxes
    for i, (box, color, label) in enumerate(zip(pred_boxes, colors, labels)):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.15,
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"{label} (IoU={ious[i]:.2f})", color=color, fontsize=9)

    ax.set_title("Intersection over Union (IoU)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("15_iou_visualization.png", dpi=100)
    print("Saved IoU visualization to 15_iou_visualization.png")

    for label, iou in zip(labels, ious):
        print(f"  {label}: IoU = {iou:.3f}")
    print("\nRule of thumb: IoU > 0.5 = correct detection")


def explain_nms():
    """Show why Non-Maximum Suppression (NMS) is needed."""
    print("\n── Non-Maximum Suppression (NMS) ──\n")

    # Multiple overlapping boxes (detector firing multiple times for same object)
    boxes = torch.tensor([
        [100, 100, 300, 250],
        [105, 102, 305, 252],
        [98,  98,  298, 248],
        [110, 110, 310, 260],
        [400, 300, 550, 420],  # different object
    ], dtype=torch.float32)

    scores = torch.tensor([0.95, 0.88, 0.82, 0.75, 0.91])

    # Apply NMS
    keep = nms(boxes, scores, iou_threshold=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, kept_only, title in [
        (axes[0], False, "Before NMS (5 boxes)"),
        (axes[1], True,  "After NMS (2 boxes)"),
    ]:
        ax.set_xlim(0, 650)
        ax.set_ylim(0, 500)
        ax.invert_yaxis()
        ax.set_title(title)

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if kept_only and i not in keep:
                continue
            color = "green" if i in keep else "red"
            alpha = 0.8 if i in keep else 0.3
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.15,
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 3, f"{score:.2f}", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig("15_nms_visualization.png", dpi=100)
    print("Saved NMS visualization to 15_nms_visualization.png")
    print(f"Before NMS: {len(boxes)} boxes")
    print(f"After NMS:  {len(keep)} boxes  (kept indices: {keep.tolist()})")


# ── Part 2: Pretrained Faster R-CNN ──────────────────────────────────────

def create_synthetic_image():
    """Create a simple synthetic test image with labeled regions."""
    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)

    # Draw simple "objects"
    draw.rectangle([50, 50, 200, 180], fill=(180, 100, 100), outline=(100, 50, 50), width=3)
    draw.text((110, 105), "OBJ1", fill="white")

    draw.ellipse([300, 100, 480, 280], fill=(100, 180, 100), outline=(50, 100, 50), width=3)
    draw.text((370, 180), "OBJ2", fill="white")

    draw.rectangle([480, 300, 600, 420], fill=(100, 100, 200), outline=(50, 50, 100), width=3)
    draw.text((525, 355), "OBJ3", fill="white")

    draw.polygon([(150, 320), (250, 250), (350, 320), (300, 430), (200, 430)],
                 fill=(220, 180, 100), outline=(150, 120, 50), width=2)
    draw.text((230, 360), "OBJ4", fill="white")

    return img


def run_pretrained_detector():
    """Run Faster R-CNN (COCO pretrained) on a test image."""
    print("\n── Pretrained Faster R-CNN ──\n")

    # Load model (downloads ~170MB on first run)
    print("Loading Faster R-CNN (pretrained on COCO)...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create or load test image
    img = create_synthetic_image()
    img.save("15_test_image.png")
    print("Created test image: 15_test_image.png")

    # Preprocess
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).to(DEVICE)

    # Inference
    print("Running detection...")
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes  = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    # Filter by confidence
    threshold = 0.5
    mask = scores >= threshold
    boxes_f  = boxes[mask]
    labels_f = labels[mask]
    scores_f = scores[mask]

    print(f"\nDetections (score >= {threshold}):")
    if len(boxes_f) == 0:
        print("  No high-confidence detections (model expects natural images)")
        # Show all predictions anyway
        mask = scores >= 0.1
        boxes_f  = boxes[mask]
        labels_f = labels[mask]
        scores_f = scores[mask]
        print(f"  Showing {len(boxes_f)} detections with score >= 0.1")

    for box, label, score in zip(boxes_f, labels_f, scores_f):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        print(f"  {class_name:15s} — score: {score:.3f}, box: [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(boxes_f), 1)))

    for i, (box, label, score) in enumerate(zip(boxes_f, labels_f, scores_f)):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        color = cmap[i % len(cmap)]
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 4, f"{class_name} {score:.2f}",
                color=color, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6))

    ax.set_title("Faster R-CNN Detection Results")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("15_detection_results.png", dpi=100)
    print("\nSaved detection results to 15_detection_results.png")
    return model


# ── Part 3: Faster R-CNN Architecture Explanation ────────────────────────

def explain_architecture():
    print("\n── How Faster R-CNN Works ──\n")
    print("""
  Input Image
       │
       ▼
  ┌─────────────────────────────┐
  │  Backbone (ResNet-50 + FPN) │  ← Extracts feature maps at multiple scales
  └─────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │  Region Proposal Network    │  ← Proposes ~2000 candidate regions ("where to look")
  │  (RPN)                      │    Uses anchor boxes at different scales/ratios
  └─────────────────────────────┘
       │ ~300 proposals (after NMS)
       ▼
  ┌─────────────────────────────┐
  │  RoI Pooling / Align        │  ← Crops and resizes each proposal to fixed size
  └─────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │  Detection Head             │  → Class probabilities (80 COCO classes)
  │                             │  → Bounding box refinement
  └─────────────────────────────┘

  Key idea: Two-stage detector
    Stage 1: RPN asks "is there an object here?" (objectness score)
    Stage 2: Head asks "what class?" and "where exactly?"
  """)

    print("Architecture comparison:")
    print("  Faster R-CNN  — high accuracy, slower (~5 FPS)")
    print("  YOLO          — real-time speed (~30-100 FPS), slightly less accurate")
    print("  SSD           — middle ground")
    print("  DETR          — Transformer-based, no anchors or NMS needed")


def main():
    print("=== Object Detection ===")
    print(f"Device: {DEVICE}\n")

    # Concepts first
    explain_iou()
    explain_nms()
    explain_architecture()

    # Run pretrained detector
    run_pretrained_detector()

    print(f"\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Bounding boxes — [x_min, y_min, x_max, y_max] format")
    print("  ✓ IoU — metric for measuring box overlap")
    print("  ✓ NMS — removing duplicate detections")
    print("  ✓ Faster R-CNN — two-stage detection pipeline")
    print("  ✓ Pretrained weights — COCO-trained model, 80 classes")
    print("\nNext steps:")
    print("  - Try YOLO (ultralytics): pip install ultralytics")
    print("  - Fine-tune on your own dataset with Roboflow")
    print("  - Explore DETR (transformer-based detection)")


if __name__ == "__main__":
    main()
