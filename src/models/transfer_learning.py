"""
transfer_learning.py
--------------------
Manages the two-phase transfer learning strategy:

  Phase 1 — Feature Extraction
    Freeze all VGG-16 layers, train only the custom head.
    Use a higher learning rate (e.g., 1e-4).

  Phase 2 — Fine-Tuning
    Unfreeze the last N layers of VGG-16, train end-to-end.
    Use a much lower learning rate (e.g., 1e-5) to avoid destroying
    the pre-trained weights.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from typing import Optional

from src.models.vgg16_model import build_vgg16_model, compile_model
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class TransferLearningManager:
    """
    Orchestrates the two-phase transfer learning workflow.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        tl_cfg = self.config["transfer_learning"]
        train_cfg = self.config["training"]
        model_cfg = self.config.get("model_config", {})

        self.fine_tune_at: int = tl_cfg.get("fine_tune_at", 15)
        self.learning_rate: float = train_cfg.get("learning_rate", 1e-4)
        self.fine_tune_lr: float = train_cfg.get("fine_tune_learning_rate", 1e-5)
        self.num_classes: int = len(self.config["data"]["classes"])
        self.input_shape = tuple(self.config["data"]["image_size"]) + (3,)

        self.model: Optional[Model] = None

    # ------------------------------------------------------------------
    # Phase 1: Feature Extraction
    # ------------------------------------------------------------------

    def build_feature_extraction_model(
        self,
        augmentation_layer: Optional[tf.keras.Sequential] = None,
    ) -> Model:
        """
        Build and compile the model for Phase 1 (frozen VGG-16 base).

        Parameters
        ----------
        augmentation_layer : optional
            Keras augmentation layers to prepend.

        Returns
        -------
        Compiled tf.keras.Model
        """
        logger.info("=== Phase 1: Feature Extraction ===")
        self.model = build_vgg16_model(
            num_classes=self.num_classes,
            input_shape=self.input_shape,
            freeze_base=True,
            include_augmentation=(augmentation_layer is not None),
            augmentation_layer=augmentation_layer,
        )
        self.model = compile_model(
            self.model,
            learning_rate=self.learning_rate,
            optimizer_name=self.config["training"].get("optimizer", "adam"),
        )
        self._log_layer_status()
        return self.model

    # ------------------------------------------------------------------
    # Phase 2: Fine-Tuning
    # ------------------------------------------------------------------

    def prepare_fine_tuning(self, model: Optional[Model] = None) -> Model:
        """
        Unfreeze VGG-16 layers from ``fine_tune_at`` onward and recompile
        with a lower learning rate.

        Parameters
        ----------
        model : tf.keras.Model, optional
            If provided, use this model; otherwise use self.model.

        Returns
        -------
        Recompiled tf.keras.Model ready for fine-tuning.
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise RuntimeError("No model available. Call build_feature_extraction_model() first.")

        logger.info("=== Phase 2: Fine-Tuning (unfreeze from layer %d) ===", self.fine_tune_at)

        # Find the VGG16 base model layer
        base_model = self._get_base_model()
        base_model.trainable = True

        # Freeze layers before fine_tune_at
        for layer in base_model.layers[: self.fine_tune_at]:
            layer.trainable = False

        trainable_layers = [l.name for l in base_model.layers if l.trainable]
        logger.info("Fine-tuning layers: %s", trainable_layers)

        # Recompile with lower learning rate
        self.model = compile_model(
            self.model,
            learning_rate=self.fine_tune_lr,
            optimizer_name=self.config["training"].get("optimizer", "adam"),
        )
        self._log_layer_status()
        return self.model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_base_model(self) -> tf.keras.Model:
        """Retrieve the VGG16 sub-model from the functional model."""
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name.lower():
                return layer
        raise ValueError("Could not find VGG16 base model inside the model graph.")

    def _log_layer_status(self) -> None:
        """Log trainable status of all layers."""
        logger.info("Layer trainability:")
        for layer in self.model.layers:
            logger.info("  %-40s trainable=%s", layer.name, layer.trainable)

    def get_model(self) -> Model:
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        return self.model
