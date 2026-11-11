"""Model evaluation module: generates accuracy/loss curves, confusion matrix,

and classification metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config


def plot_training_history(history_dict: dict, save_path: Path = config.TRAINING_HISTORY_PATH):
    """Plot training and validation accuracy and loss curves side-by-side."""
    if not history_dict:
        return

    epochs_range = range(1, len(history_dict.get("accuracy", [])) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    if "accuracy" in history_dict:
        plt.plot(epochs_range, history_dict["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history_dict:
        plt.plot(epochs_range, history_dict["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)

    # Loss subplot
    plt.subplot(1, 2, 2)
    if "loss" in history_dict:
        plt.plot(epochs_range, history_dict["loss"], "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history_dict:
        plt.plot(epochs_range, history_dict["val_loss"], "m--s", label="Validation Loss", linewidth=2)
    plt.title("Model Loss Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Categorical Crossentropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved training curves to: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    save_path: Path = config.CONFUSION_MATRIX_PATH,
):
    """Compute and render a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    display_labels = [config.DISPLAY_NAMES.get(name, name) for name in class_names]

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray",
    )
    plt.title("Confusion Matrix (Test Set)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Predicted Object Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved confusion matrix to: {save_path}")


def evaluate_model(
    model_path: Path = config.BEST_MODEL_PATH,
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    history_dict: Optional[dict] = None,
) -> dict:
    """Evaluate trained model on the independent test split and generate all artifacts."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    # Load splits manifest if paths not provided
    if test_paths is None or test_labels is None:
        manifest_file = config.REPORTS_DIR / "data_splits.json"
        if not manifest_file.exists():
            raise FileNotFoundError("Data splits manifest not found. Cannot evaluate test set.")
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        test_paths = [item["path"] for item in manifest.get("test", [])]
        test_labels = [item["label"] for item in manifest.get("test", [])]

    if not test_paths:
        raise ValueError("Test set is empty. Cannot evaluate.")

    print(f"Evaluating {len(test_paths)} test images...")
    model = keras.models.load_model(str(model_path))

    # Preprocess all test images
    batch_images = []
    valid_y_true = []
    for p, lbl in zip(test_paths, test_labels):
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB").resize(config.IMG_SIZE, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32)
                arr = preprocess_input(arr)
                batch_images.append(arr)
                valid_y_true.append(lbl)
        except Exception as e:
            print(f"  [Warning] Skipping test image {p}: {e}")

    X_test = np.array(batch_images)
    y_true = np.array(valid_y_true)

    # Predict
    raw_preds = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_pred = np.argmax(raw_preds, axis=1)

    # Calculate actual test metrics
    test_accuracy = float(np.mean(y_true == y_pred))

    # Classification report
    class_report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    class_report_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=[config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES],
        zero_division=0,
    )

    # Save text report
    with config.CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write(class_report_str)

    # Save JSON metrics
    metrics_summary = {
        "test_accuracy": test_accuracy,
        "test_sample_count": len(y_true),
        "macro_avg": class_report_dict.get("macro avg", {}),
        "weighted_avg": class_report_dict.get("weighted avg", {}),
        "per_class": {
            cls_name: class_report_dict.get(cls_name, {})
            for cls_name in config.CLASS_NAMES
        },
    }
    with config.METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Generate charts
    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    if history_dict:
        plot_training_history(history_dict, config.TRAINING_HISTORY_PATH)

    print("\n--- TEST EVALUATION REPORT ---")
    print(class_report_str)
    print(f"Actual Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Saved artifacts to {config.REPORTS_DIR}")

    return metrics_summary


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
