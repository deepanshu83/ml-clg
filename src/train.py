"""Model training script with transfer learning and two-stage fine-tuning."""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src import config
from src.dataset import audit_and_clean_dataset, prepare_datasets
from src.evaluate import evaluate_model
from src.model import build_mobilenetv2_model, unfreeze_for_fine_tuning


def train_classifier(
    initial_epochs: int = config.INITIAL_EPOCHS,
    fine_tune_epochs: int = config.FINE_TUNE_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    enable_fine_tuning: bool = True,
) -> dict:
    """Execute transfer learning training workflow on the 10-class object dataset."""
    print("=" * 70)
    print("  STARTING MODEL TRAINING PIPELINE (MobileNetV2)")
    print("=" * 70)

    # 1. Audit dataset
    stats = audit_and_clean_dataset()
    total_valid = stats["total_valid"]
    print(f"Total valid images verified: {total_valid}")
    if total_valid < 50:
        raise ValueError(
            f"Insufficient dataset images found ({total_valid} total). "
            f"Please run 'python scripts/download_images.py --per-class 50' first."
        )

    for cat_name, info in stats["categories"].items():
        print(f"  - {cat_name:<16}: {info['valid']} valid images")

    # 2. Build datasets (70% train, 15% val, 15% test)
    print("\n[Step 1/4] Preparing stratified data splits and tf.data pipelines...")
    train_ds, val_ds, test_ds, splits = prepare_datasets(batch_size=batch_size)
    print(f"  Training set:   {len(splits['train'][0])} images")
    print(f"  Validation set: {len(splits['val'][0])} images")
    print(f"  Test set:       {len(splits['test'][0])} images")

    # 3. Build model
    print("\n[Step 2/4] Constructing MobileNetV2 Transfer Learning Architecture...")
    model = build_mobilenetv2_model()
    model.summary(print_fn=lambda x: print("  " + x))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 4. Phase 1: Train classification head with frozen backbone
    print(f"\n[Step 3/4] Phase 1: Training Classification Head ({initial_epochs} epochs)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = {k: list(v) for k, v in history_phase1.history.items()}

    # 5. Phase 2: Optional Fine-tuning of top layers
    if enable_fine_tuning and fine_tune_epochs > 0:
        print(f"\n[Step 4/4] Phase 2: Fine-Tuning Top Backbone Layers ({fine_tune_epochs} epochs)...")
        # Load the best weights from Phase 1 before unfreezing
        if config.BEST_MODEL_PATH.exists():
            model = keras.models.load_model(str(config.BEST_MODEL_PATH))

        model = unfreeze_for_fine_tuning(model, unfreeze_layers_count=30, learning_rate=config.FINE_TUNE_LR)

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=len(history_phase1.epoch),
            callbacks=callbacks,
            verbose=1,
        )

        for k, v in history_phase2.history.items():
            if k in combined_history:
                combined_history[k].extend(v)
            else:
                combined_history[k] = list(v)

    # 6. Evaluate final best model on independent test set
    print("\nRunning comprehensive evaluation on independent 15% test set...")
    eval_metrics = evaluate_model(
        model_path=config.BEST_MODEL_PATH,
        test_paths=splits["test"][0],
        test_labels=splits["test"][1],
        history_dict=combined_history,
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print(f"  Best Model Saved: {config.BEST_MODEL_PATH}")
    print(f"  Test Accuracy:    {eval_metrics.get('test_accuracy', 0.0) * 100:.2f}%")
    print(f"  Reports Generated in: {config.REPORTS_DIR}")
    print("=" * 70)

    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 classifier on 10 object classes.")
    parser.add_argument("--epochs", type=int, default=config.INITIAL_EPOCHS, help="Phase 1 initial epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help="Phase 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--no-fine-tune", action="store_true", help="Disable fine-tuning phase")
    args = parser.parse_args()

    train_classifier(
        initial_epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        enable_fine_tuning=not args.no_fine_tune,
    )


if __name__ == "__main__":
    main()
