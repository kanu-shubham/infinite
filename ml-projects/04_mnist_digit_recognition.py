"""
Project 4: Handwritten Digit Recognition (MNIST)
==================================================
Recognize handwritten digits (0-9) from 28x28 pixel images.

What you'll learn:
- Working with image data
- Building a neural network with TensorFlow/Keras
- Training with epochs and batch size
- Evaluating with accuracy and confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def run_sklearn_version():
    """Fallback using scikit-learn's smaller digits dataset (8x8 images)."""
    print("Using scikit-learn digits dataset (8x8 images)\n")

    # ── 1. Load data ─────────────────────────────────────────────────────
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Samples: {len(X)}")
    print(f"Image shape: 8x8 pixels")
    print(f"Classes: 0-9\n")

    # Visualize sample digits
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    plt.suptitle("Sample Digits (8x8)", fontsize=14)
    plt.tight_layout()
    plt.savefig("04_mnist_samples.png", dpi=100)
    print("Saved sample digits to 04_mnist_samples.png")

    # ── 2. Prepare data ──────────────────────────────────────────────────
    X_normalized = X / 16.0  # 8x8 digits have max value 16

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Train neural network ──────────────────────────────────────────
    print("Training MLPClassifier (neural network)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train, y_train)

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nAccuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Digit Recognition Confusion Matrix ({accuracy:.2%})")
    plt.tight_layout()
    plt.savefig("04_mnist_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 04_mnist_confusion_matrix.png")

    # Training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_mnist_loss_curve.png", dpi=100)
    print("Saved loss curve to 04_mnist_loss_curve.png")


def run_tensorflow_version():
    """Full version using TensorFlow/Keras with MNIST (28x28 images)."""
    print("Using TensorFlow with full MNIST dataset (28x28 images)\n")

    # ── 1. Load data ─────────────────────────────────────────────────────
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Classes: 0-9\n")

    # Visualize sample digits
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap="gray")
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis("off")
    plt.suptitle("Sample MNIST Digits (28x28)", fontsize=14)
    plt.tight_layout()
    plt.savefig("04_mnist_samples.png", dpi=100)
    print("Saved sample digits to 04_mnist_samples.png")

    # ── 2. Prepare data ──────────────────────────────────────────────────
    X_train_norm = X_train.astype("float32") / 255.0
    X_test_norm = X_test.astype("float32") / 255.0

    # ── 3. Build the model ───────────────────────────────────────────────
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Model Summary:")
    model.summary()
    print()

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("Training...")
    history = model.fit(
        X_train_norm, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f}\n")

    y_pred = np.argmax(model.predict(X_test_norm, verbose=0), axis=1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_mnist_training_history.png", dpi=100)
    print("Saved training history to 04_mnist_training_history.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"MNIST Confusion Matrix ({test_acc:.2%})")
    plt.tight_layout()
    plt.savefig("04_mnist_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 04_mnist_confusion_matrix.png")


def main():
    print("=== Handwritten Digit Recognition ===\n")

    if HAS_TF:
        run_tensorflow_version()
    else:
        print("TensorFlow not installed. Using scikit-learn fallback.")
        print("Install TensorFlow for the full experience: pip install tensorflow\n")
        run_sklearn_version()


if __name__ == "__main__":
    main()
