"""Dataset loading, preprocessing, deduplication, and data pipeline module."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config
from scripts.download_images import compute_dhash, is_near_duplicate


def audit_and_clean_dataset(
    dataset_dir: Path = config.DATASET_DIR,
    auto_remove_corrupt: bool = False,
) -> Dict:
    """Inspect dataset, detect corrupted images, identify perceptual duplicates,

    and return comprehensive statistics.
    """
    stats = {
        "categories": {},
        "total_valid": 0,
        "total_corrupt": 0,
        "total_duplicates": 0,
        "corrupt_files": [],
        "duplicate_files": [],
    }

    for class_name in config.CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            stats["categories"][class_name] = {
                "valid": 0,
                "corrupt": 0,
                "duplicates": 0,
                "target": config.TARGET_PER_CLASS,
            }
            continue

        valid_paths = []
        known_hashes = set()
        cat_corrupt = 0
        cat_dupes = 0

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file() or img_path.name.startswith("."):
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            # Check image integrity
            try:
                with Image.open(img_path) as test_img:
                    test_img.verify()

                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < 100 or h < 100:
                        raise ValueError(f"Image too small ({w}x{h})")
                    rgb_img = img.convert("RGB")
                    dhash = compute_dhash(rgb_img)

                    # Check for perceptual duplicate
                    if is_near_duplicate(dhash, known_hashes, threshold=5):
                        cat_dupes += 1
                        stats["duplicate_files"].append(str(img_path))
                        continue

                    known_hashes.add(dhash)
                    valid_paths.append(img_path)

            except Exception as e:
                cat_corrupt += 1
                stats["corrupt_files"].append((str(img_path), str(e)))
                if auto_remove_corrupt:
                    try:
                        img_path.unlink()
                    except OSError:
                        pass

        stats["categories"][class_name] = {
            "valid": len(valid_paths),
            "corrupt": cat_corrupt,
            "duplicates": cat_dupes,
            "target": config.TARGET_PER_CLASS,
        }
        stats["total_valid"] += len(valid_paths)
        stats["total_corrupt"] += cat_corrupt
        stats["total_duplicates"] += cat_dupes

    return stats


def get_labeled_image_paths(dataset_dir: Path = config.DATASET_DIR) -> Tuple[List[str], List[int]]:
    """Collect valid image file paths and numeric class labels."""
    file_paths = []
    labels = []
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    for class_name in config.CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        label_idx = class_to_idx[class_name]
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    with Image.open(p) as img:
                        img.verify()
                    file_paths.append(str(p))
                    labels.append(label_idx)
                except Exception:
                    continue

    return file_paths, labels


def split_dataset(
    file_paths: List[str],
    labels: List[int],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Split data into 70% Train, 15% Validation, and 15% Test sets using stratified sampling.

    Splitting is executed on raw image paths prior to any augmentation to strictly prevent data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # First split off test set (15%)
    val_plus_test_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=val_plus_test_ratio,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    # Then split temp into val and test sets (50/50 of remaining 30% -> 15% each)
    relative_test_ratio = test_ratio / val_plus_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )

    split_manifest = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    # Save manifest for reproducibility and audit
    manifest_data = {
        split_name: [{"path": p, "label": int(l), "class": config.CLASS_NAMES[l]} for p, l in zip(paths, lbls)]
        for split_name, (paths, lbls) in split_manifest.items()
    }
    manifest_file = config.REPORTS_DIR / "data_splits.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return split_manifest


def build_augmentation_layer() -> keras.Sequential:
    """Create data augmentation pipeline with rotation, zoom, brightness, and horizontal flip."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_random_flip"),
            layers.RandomRotation(0.15, name="aug_random_rotation"),
            layers.RandomZoom(0.15, name="aug_random_zoom"),
            layers.RandomBrightness(factor=0.15, name="aug_random_brightness"),
        ],
        name="data_augmentation",
    )


def load_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor, is_training: bool = False, augment_model=None):
    """Load image from disk, convert to RGB, resize to 224x224, apply augmentation if training,

    and apply MobileNetV2 preprocessing.
    """
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if is_training and augment_model is not None:
        img = augment_model(tf.expand_dims(img, axis=0), training=True)[0]

    # Preprocess for MobileNetV2 (scales pixel values between -1.0 and 1.0)
    img = preprocess_input(img)
    one_hot_label = tf.one_hot(label, depth=len(config.CLASS_NAMES))
    return img, one_hot_label


def create_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
    augment: bool = True,
) -> tf.data.Dataset:
    """Build high-performance tf.data.Dataset pipeline."""
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    augment_model = build_augmentation_layer() if (is_training and augment) else None

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, is_training=is_training, augment_model=augment_model),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_datasets(dataset_dir: Path = config.DATASET_DIR, batch_size: int = config.BATCH_SIZE):
    """High-level helper to collect images, split cleanly, and generate train, val, and test tf.data pipelines."""
    file_paths, labels = get_labeled_image_paths(dataset_dir)
    if not file_paths:
        raise ValueError(f"No valid images found in {dataset_dir}. Please run the downloader first.")

    splits = split_dataset(file_paths, labels)

    train_ds = create_tf_dataset(splits["train"][0], splits["train"][1], batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(splits["val"][0], splits["val"][1], batch_size=batch_size, is_training=False)
    test_ds = create_tf_dataset(splits["test"][0], splits["test"][1], batch_size=batch_size, is_training=False)

    return train_ds, val_ds, test_ds, splits
