"""
model_utils.py
--------------
Helper functions for saving, loading, and inspecting Keras models.
"""

import os
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Model,
    save_path: str,
    save_format: str = "h5",
) -> None:
    """
    Save a Keras model to disk.

    Parameters
    ----------
    model : tf.keras.Model
    save_path : str
        Full path including filename (e.g., 'models/final/animal_classifier.h5').
    save_format : str
        'h5' for HDF5 or 'tf' for TensorFlow SavedModel format.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if save_format == "h5":
        model.save(save_path)
        logger.info("Model saved (HDF5): %s", save_path)
    else:
        # SavedModel format
        saved_model_dir = str(Path(save_path).with_suffix(""))
        model.save(saved_model_dir, save_format="tf")
        logger.info("Model saved (SavedModel): %s", saved_model_dir)


def load_model(model_path: str) -> Model:
    """
    Load a Keras model from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved model (.h5 file or SavedModel directory).

    Returns
    -------
    tf.keras.Model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded from: %s", model_path)
    return model


def get_model_summary(model: Model) -> str:
    """Return the model summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def count_parameters(model: Model) -> dict:
    """
    Count total, trainable, and non-trainable parameters.

    Returns
    -------
    dict with keys: total, trainable, non_trainable
    """
    trainable = int(
        sum(tf.size(w).numpy() for w in model.trainable_weights)
    )
    non_trainable = int(
        sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    )
    return {
        "total":         trainable + non_trainable,
        "trainable":     trainable,
        "non_trainable": non_trainable,
    }


def freeze_layers(model: Model, num_layers: int) -> Model:
    """
    Freeze the first ``num_layers`` layers of the model.

    Parameters
    ----------
    model : tf.keras.Model
    num_layers : int

    Returns
    -------
    Modified model (in-place).
    """
    for layer in model.layers[:num_layers]:
        layer.trainable = False
    logger.info("Froze first %d layers.", num_layers)
    return model


def unfreeze_layers(model: Model, from_layer: int) -> Model:
    """
    Unfreeze all layers from index ``from_layer`` onward.

    Parameters
    ----------
    model : tf.keras.Model
    from_layer : int

    Returns
    -------
    Modified model (in-place).
    """
    for layer in model.layers[from_layer:]:
        layer.trainable = True
    logger.info("Unfroze layers from index %d onward.", from_layer)
    return model
