"""Inference module for single-image and batch predictions."""

import argparse
import io
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config

# Cached model instance for fast repeated predictions
_CACHED_MODEL = None


def get_model(model_path: Path = config.BEST_MODEL_PATH) -> keras.Model:
    """Retrieve or load cached Keras model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please train the model before running predictions."
            )
        _CACHED_MODEL = keras.models.load_model(str(model_path))
    return _CACHED_MODEL


def preprocess_image(image_input: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
    """Load, convert to RGB, resize to 224x224, and apply MobileNetV2 preprocessing."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize(config.IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array)
    # Expand to batch dimension (1, 224, 224, 3)
    return np.expand_dims(img_preprocessed, axis=0)


def predict_image(
    image_input: Union[str, Path, bytes, Image.Image],
    model: keras.Model = None,
    top_k: int = config.TOP_K,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> Dict:
    """Run inference on an input image, returning top predictions and confidence levels."""
    if model is None:
        model = get_model()

    input_tensor = preprocess_image(image_input)
    raw_probs = model.predict(input_tensor, verbose=0)[0]

    # Sort descending
    sorted_indices = np.argsort(raw_probs)[::-1]
    top_indices = sorted_indices[:top_k]

    best_idx = top_indices[0]
    best_class = config.CLASS_NAMES[best_idx]
    best_conf = float(raw_probs[best_idx])
    best_display = config.DISPLAY_NAMES.get(best_class, best_class)

    top_predictions = []
    for idx in top_indices:
        cls_name = config.CLASS_NAMES[idx]
        prob = float(raw_probs[idx])
        top_predictions.append(
            {
                "class": cls_name,
                "display_name": config.DISPLAY_NAMES.get(cls_name, cls_name),
                "confidence": prob,
                "confidence_percent": round(prob * 100, 2),
            }
        )

    is_low_confidence = best_conf < confidence_threshold
    warning_msg = None
    if is_low_confidence:
        warning_msg = (
            f"Low confidence warning: Top prediction is '{best_display}' with only {best_conf * 100:.1f}% confidence "
            f"(threshold: {confidence_threshold * 100:.0f}%). The image might be out of domain or ambiguous."
        )

    return {
        "predicted_class": best_class,
        "display_name": best_display,
        "confidence": best_conf,
        "confidence_percent": round(best_conf * 100, 2),
        "is_low_confidence": is_low_confidence,
        "warning_message": warning_msg,
        "top_predictions": top_predictions,
        "all_probabilities": {
            config.CLASS_NAMES[i]: float(raw_probs[i])
            for i in range(len(config.CLASS_NAMES))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Classify an object image using trained MobileNetV2.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    args = parser.parse_args()

    result = predict_image(args.image)
    print("=" * 60)
    print(f"Prediction: {result['display_name']} ({result['confidence_percent']}%)")
    if result["is_low_confidence"]:
        print(f"⚠️  {result['warning_message']}")
    print("-" * 60)
    print("Top Predictions:")
    for rank, p in enumerate(result["top_predictions"], 1):
        print(f"  {rank}. {p['display_name']:<16} : {p['confidence_percent']:>6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
