"""
test_model.py
-------------
Unit tests for model architecture and utilities.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vgg16_model import build_vgg16_model, compile_model
from src.models.model_utils import count_parameters, get_model_summary


class TestVGG16Model(unittest.TestCase):
    """Tests for the VGG-16 model builder."""

    @classmethod
    def setUpClass(cls):
        """Build the model once for all tests."""
        cls.model = build_vgg16_model(
            num_classes=15,
            input_shape=(224, 224, 3),
            freeze_base=True,
        )

    def test_model_output_shape(self):
        """Model output should have shape (batch, 15)."""
        dummy_input = np.zeros((2, 224, 224, 3), dtype=np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (2, 15))

    def test_output_sums_to_one(self):
        """Softmax output should sum to 1 for each sample."""
        dummy_input = np.random.rand(4, 224, 224, 3).astype(np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        sums = output.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_model_has_correct_layers(self):
        """Model should contain expected custom layers."""
        layer_names = [l.name for l in self.model.layers]
        self.assertIn("global_avg_pool", layer_names)
        self.assertIn("dense_512", layer_names)
        self.assertIn("dense_256", layer_names)
        self.assertIn("predictions", layer_names)

    def test_base_frozen(self):
        """VGG-16 base should be frozen in feature extraction mode."""
        # Find the VGG16 sub-model
        vgg_layer = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name.lower():
                vgg_layer = layer
                break
        self.assertIsNotNone(vgg_layer, "VGG16 base model not found")
        self.assertFalse(vgg_layer.trainable, "VGG16 base should be frozen")

    def test_trainable_params_reduced_when_frozen(self):
        """Frozen model should have fewer trainable params than total."""
        params = count_parameters(self.model)
        self.assertLess(params["trainable"], params["total"])
        self.assertGreater(params["trainable"], 0)

    def test_model_summary_not_empty(self):
        """Model summary should return a non-empty string."""
        summary = get_model_summary(self.model)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 100)


class TestModelCompilation(unittest.TestCase):
    """Tests for model compilation."""

    def test_compile_with_adam(self):
        """Model should compile with Adam optimizer."""
        model = build_vgg16_model(num_classes=15, freeze_base=True)
        compiled = compile_model(model, learning_rate=0.0001, optimizer_name="adam")
        self.assertIsNotNone(compiled.optimizer)
        self.assertIsInstance(compiled.optimizer, tf.keras.optimizers.Adam)

    def test_compile_with_sgd(self):
        """Model should compile with SGD optimizer."""
        model = build_vgg16_model(num_classes=15, freeze_base=True)
        compiled = compile_model(model, learning_rate=0.001, optimizer_name="sgd")
        self.assertIsInstance(compiled.optimizer, tf.keras.optimizers.SGD)

    def test_model_has_loss(self):
        """Compiled model should have categorical crossentropy loss."""
        model = build_vgg16_model(num_classes=15, freeze_base=True)
        compiled = compile_model(model)
        self.assertIsNotNone(compiled.loss)


class TestFineTuning(unittest.TestCase):
    """Tests for fine-tuning layer unfreezing."""

    def test_unfreeze_layers(self):
        """Fine-tuning should make some VGG-16 layers trainable."""
        from src.models.model_utils import unfreeze_layers

        model = build_vgg16_model(num_classes=15, freeze_base=True)

        # Find VGG16 base
        vgg_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name.lower():
                vgg_layer = layer
                break

        self.assertIsNotNone(vgg_layer)
        vgg_layer.trainable = True

        # Freeze first 15 layers
        for layer in vgg_layer.layers[:15]:
            layer.trainable = False

        trainable_vgg = [l for l in vgg_layer.layers if l.trainable]
        self.assertGreater(len(trainable_vgg), 0)


if __name__ == "__main__":
    unittest.main()
