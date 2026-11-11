"""MobileNetV2 Transfer Learning Model Architecture and Fine-tuning Utilities."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_mobilenetv2_model(
    num_classes: int = len(config.CLASS_NAMES),
    input_shape: tuple = config.INPUT_SHAPE,
    dense_units: int = 256,
    dropout_rate: float = 0.4,
    learning_rate: float = config.INITIAL_LR,
) -> keras.Model:
    """Construct MobileNetV2 transfer learning architecture with a custom classification head.

    Initially, the base model backbone weights are frozen to train the newly added dense layers.
    """
    # Pre-trained MobileNetV2 backbone (ImageNet weights)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze backbone for Phase 1

    # Define model architecture
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_object_classifier")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_for_fine_tuning(
    model: keras.Model,
    unfreeze_layers_count: int = 30,
    learning_rate: float = config.FINE_TUNE_LR,
) -> keras.Model:
    """Unfreeze top layers of the MobileNetV2 backbone for Stage 2 fine-tuning

    using a lower learning rate. BatchNormalization layers remain frozen to preserve running statistics.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        # Check if backbone layers are directly in model
        layers_to_unfreeze = [l for l in model.layers if not isinstance(l, layers.BatchNormalization)][-unfreeze_layers_count:]
        for l in layers_to_unfreeze:
            l.trainable = True
    else:
        base_model.trainable = True
        # Freeze all layers except the top unfreeze_layers_count
        for layer in base_model.layers[:-unfreeze_layers_count]:
            layer.trainable = False
        # Keep BatchNormalization frozen
        for layer in base_model.layers[-unfreeze_layers_count:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    # Re-compile with smaller learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
