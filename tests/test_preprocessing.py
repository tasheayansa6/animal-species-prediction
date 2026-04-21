"""
test_preprocessing.py
---------------------
Unit tests for preprocessing and augmentation functions.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import VALID_CLASSES
from src.data.augmentation import DataAugmentor


class TestValidClasses(unittest.TestCase):
    """Tests for the 15-class English class set."""

    def test_all_fifteen_classes_present(self):
        self.assertEqual(len(VALID_CLASSES), 15)

    def test_specific_classes(self):
        expected = {
            "Beetle", "Butterfly", "Cat", "Cow", "Dog",
            "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
            "Mouse", "Panda", "Spider", "Tiger", "Zebra",
        }
        self.assertEqual(VALID_CLASSES, expected)

    def test_classes_are_unique(self):
        self.assertEqual(len(VALID_CLASSES), len(set(VALID_CLASSES)))


class TestDataAugmentor(unittest.TestCase):
    """Tests for DataAugmentor class."""

    def setUp(self):
        self.augmentor = DataAugmentor()

    def test_get_train_datagen(self):
        """Train datagen should have augmentation parameters set."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = self.augmentor.get_train_datagen()
        self.assertIsInstance(datagen, ImageDataGenerator)
        self.assertEqual(datagen.rescale, 1.0 / 255)
        self.assertGreater(datagen.rotation_range, 0)

    def test_get_eval_datagen(self):
        """Eval datagen should only rescale, no augmentation."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = self.augmentor.get_eval_datagen()
        self.assertIsInstance(datagen, ImageDataGenerator)
        self.assertEqual(datagen.rescale, 1.0 / 255)
        self.assertEqual(datagen.rotation_range, 0)
        self.assertFalse(datagen.horizontal_flip)

    def test_get_augmentation_layers(self):
        """Augmentation layers should be a Sequential model."""
        import tensorflow as tf
        aug_layers = self.augmentor.get_augmentation_layers()
        self.assertIsInstance(aug_layers, tf.keras.Sequential)
        self.assertGreater(len(aug_layers.layers), 0)

    def test_augment_image_output_shape(self):
        """Augmented image should have the same shape as input."""
        import tensorflow as tf
        image = tf.random.uniform((224, 224, 3))
        label = tf.constant(0)
        aug_image, aug_label = DataAugmentor.augment_image(image, label)
        self.assertEqual(aug_image.shape, image.shape)

    def test_augment_image_values_in_range(self):
        """Augmented pixel values should remain in [0, 1]."""
        import tensorflow as tf
        image = tf.random.uniform((224, 224, 3))
        label = tf.constant(0)
        aug_image, _ = DataAugmentor.augment_image(image, label)
        self.assertGreaterEqual(float(tf.reduce_min(aug_image)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(aug_image)), 1.0)

    def test_custom_config(self):
        """Custom config should override defaults."""
        custom_cfg = {"rotation_range": 45, "horizontal_flip": False}
        augmentor = DataAugmentor(aug_config=custom_cfg)
        datagen = augmentor.get_train_datagen()
        self.assertEqual(datagen.rotation_range, 45)
        self.assertFalse(datagen.horizontal_flip)


class TestImageResizing(unittest.TestCase):
    """Tests for image resizing logic."""

    def test_resize_large_image(self):
        """Large images should be resized to 224×224."""
        img = Image.new("RGB", (1024, 768))
        resized = img.resize((224, 224), Image.BILINEAR)
        self.assertEqual(resized.size, (224, 224))

    def test_resize_small_image(self):
        """Small images should be upscaled to 224×224."""
        img = Image.new("RGB", (50, 50))
        resized = img.resize((224, 224), Image.BILINEAR)
        self.assertEqual(resized.size, (224, 224))

    def test_rgba_to_rgb_conversion(self):
        """RGBA images should be converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        rgb = img.convert("RGB")
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(len(rgb.getbands()), 3)

    def test_grayscale_to_rgb(self):
        """Grayscale images should be converted to RGB (3 channels)."""
        img = Image.new("L", (100, 100), color=128)
        rgb = img.convert("RGB")
        arr = np.array(rgb)
        self.assertEqual(arr.shape[2], 3)

    def test_pixel_normalization(self):
        """Pixel values should be in [0, 1] after dividing by 255."""
        img = Image.new("RGB", (224, 224), color=(200, 100, 50))
        arr = np.array(img, dtype=np.float32) / 255.0
        self.assertGreaterEqual(arr.min(), 0.0)
        self.assertLessEqual(arr.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
