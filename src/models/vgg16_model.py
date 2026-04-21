"""
vgg16_model.py
--------------
Builds the VGG-16 based animal classifier.

Architecture:
  VGG-16 (ImageNet weights, top removed)
    └── GlobalAveragePooling2D
    └── Dense(512, relu) + BatchNorm + Dropout(0.5)
    └── Dense(256, relu) + BatchNorm + Dropout(0.3)
    └── Dense(15, softmax)
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Flatten,
)
from tensorflow.keras.regularizers import l2
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

NUM_CLASSES = 15
INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 3)


def build_vgg16_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    freeze_base: bool = True,
    l2_lambda: float = 0.0001,
    dropout_1: float = 0.5,
    dropout_2: float = 0.3,
    include_augmentation: bool = False,
    augmentation_layer: Optional[tf.keras.Sequential] = None,
) -> Model:
    """
    Build and return the VGG-16 transfer learning model.

    Parameters
    ----------
    num_classes : int
        Number of output classes (15 for the Animal Image Classification dataset).
    input_shape : tuple
        Input image shape (H, W, C). Must be (224, 224, 3) for VGG-16.
    freeze_base : bool
        If True, freeze all VGG-16 convolutional layers (feature extraction mode).
        Set to False for fine-tuning.
    l2_lambda : float
        L2 regularization factor for Dense layers.
    dropout_1 : float
        Dropout rate after the first Dense layer.
    dropout_2 : float
        Dropout rate after the second Dense layer.
    include_augmentation : bool
        If True, prepend the augmentation_layer to the model.
    augmentation_layer : tf.keras.Sequential, optional
        Augmentation layers to prepend (only used when include_augmentation=True).

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.
    """
    # ---- Base model ----
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_base

    if freeze_base:
        logger.info("VGG-16 base frozen — feature extraction mode.")
    else:
        logger.info("VGG-16 base trainable — fine-tuning mode.")

    # ---- Build functional model ----
    inputs = Input(shape=input_shape, name="input_image")
    x = inputs

    # Optional augmentation (active only during training)
    if include_augmentation and augmentation_layer is not None:
        x = augmentation_layer(x, training=True)

    # VGG-16 feature extractor
    x = base_model(x, training=False)  # training=False keeps BN layers frozen

    # Custom classification head
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = Dense(512, activation="relu", kernel_regularizer=l2(l2_lambda), name="dense_512")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(dropout_1, name="dropout_1")(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(l2_lambda), name="dense_256")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(dropout_2, name="dropout_2")(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="VGG16_AnimalClassifier")

    logger.info(
        "Model built — total params: %s | trainable: %s",
        f"{model.count_params():,}",
        f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}",
    )
    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.0001,
    optimizer_name: str = "adam",
) -> Model:
    """
    Compile the model with the given optimizer and learning rate.

    Parameters
    ----------
    model : tf.keras.Model
    learning_rate : float
    optimizer_name : str
        One of 'adam', 'sgd', 'rmsprop'.

    Returns
    -------
    Compiled tf.keras.Model
    """
    optimizers = {
        "adam":    tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd":     tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    }
    optimizer = optimizers.get(optimizer_name.lower(), optimizers["adam"])

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    logger.info("Model compiled with %s (lr=%.6f)", optimizer_name, learning_rate)
    return model
