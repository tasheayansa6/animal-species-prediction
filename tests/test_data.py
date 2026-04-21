"""
test_data.py
------------
Unit tests for data loading and dataset utilities.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import AnimalDataset
from src.utils.config import load_config


class TestAnimalDataset(unittest.TestCase):
    """Tests for AnimalDataset class."""

    CONFIG_PATH = "config/config.yaml"

    def test_class_names_count(self):
        """Dataset should have exactly 15 classes."""
        self.assertEqual(len(AnimalDataset.CLASS_NAMES), 15)

    def test_class_names_content(self):
        """Verify all expected class names are present."""
        expected = {
            "Beetle", "Butterfly", "Cat", "Cow", "Dog",
            "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
            "Mouse", "Panda", "Spider", "Tiger", "Zebra",
        }
        self.assertEqual(set(AnimalDataset.CLASS_NAMES), expected)

    def test_config_loading(self):
        """Config should load without errors."""
        if not os.path.exists(self.CONFIG_PATH):
            self.skipTest("Config file not found")
        dataset = AnimalDataset(config_path=self.CONFIG_PATH)
        self.assertIsNotNone(dataset.config)
        self.assertEqual(dataset.image_size, (224, 224))

    def test_missing_directory_raises(self):
        """build_generators should raise FileNotFoundError for missing dirs."""
        if not os.path.exists(self.CONFIG_PATH):
            self.skipTest("Config file not found")
        dataset = AnimalDataset(config_path=self.CONFIG_PATH)
        dataset.train_dir = "/nonexistent/path"
        with self.assertRaises(FileNotFoundError):
            dataset.build_generators()


class TestDataPreprocessor(unittest.TestCase):
    """Tests for DataPreprocessor class."""

    def setUp(self):
        """Create a temporary directory structure for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.test_dir) / "src" / "data"

        # Mimic the real dataset structure with English capitalized class folders
        test_classes = ["Cat", "Dog", "Tiger"]
        for split in ["Training Data", "Validation Data", "Testing Data"]:
            for cls in test_classes:
                cls_dir = self.raw_dir / split / cls
                cls_dir.mkdir(parents=True)
                for i in range(5):
                    img = Image.new("RGB", (300, 300), color=(i * 50, i * 30, i * 20))
                    img.save(cls_dir / f"img_{i:03d}.jpg")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_italian_to_english_mapping(self):
        """VALID_CLASSES should contain all 15 expected capitalized class names."""
        from src.data.preprocessing import VALID_CLASSES
        expected = {
            "Beetle", "Butterfly", "Cat", "Cow", "Dog",
            "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
            "Mouse", "Panda", "Spider", "Tiger", "Zebra",
        }
        self.assertEqual(VALID_CLASSES, expected)

    def test_collect_images(self):
        """Should collect images from capitalized English-named folders."""
        from src.data.preprocessing import DataPreprocessor

        with patch.object(DataPreprocessor, "__init__", lambda self, *a, **kw: None):
            preprocessor = DataPreprocessor.__new__(DataPreprocessor)
            preprocessor.raw_dir = self.raw_dir
            preprocessor.processed_dir = Path(self.test_dir) / "processed"
            preprocessor.image_size = (224, 224)
            preprocessor.train_split = 0.7
            preprocessor.val_split = 0.15
            preprocessor.seed = 42
            preprocessor.classes = ["Cat", "Dog", "Tiger"]

            import random
            import numpy as np
            random.seed(42)
            np.random.seed(42)

            class_image_map = preprocessor._collect_images()

        self.assertIn("Cat", class_image_map)
        self.assertIn("Dog", class_image_map)
        self.assertIn("Tiger", class_image_map)


class TestImagePreprocessing(unittest.TestCase):
    """Tests for image preprocessing utilities."""

    def test_resize_image(self):
        """Images should be resized to 224×224."""
        img = Image.new("RGB", (500, 300), color=(100, 150, 200))
        img_resized = img.resize((224, 224), Image.BILINEAR)
        self.assertEqual(img_resized.size, (224, 224))

    def test_normalize_pixel_values(self):
        """Pixel values should be in [0, 1] after normalization."""
        arr = np.array(Image.new("RGB", (224, 224), color=(128, 64, 32)), dtype=np.float32)
        arr /= 255.0
        self.assertGreaterEqual(arr.min(), 0.0)
        self.assertLessEqual(arr.max(), 1.0)

    def test_rgb_conversion(self):
        """Grayscale images should be converted to RGB."""
        gray_img = Image.new("L", (100, 100), color=128)
        rgb_img = gray_img.convert("RGB")
        self.assertEqual(rgb_img.mode, "RGB")
        self.assertEqual(len(rgb_img.getbands()), 3)


if __name__ == "__main__":
    unittest.main()
