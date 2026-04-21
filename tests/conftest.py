"""
conftest.py
-----------
Pytest configuration and shared fixtures for all test modules.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def class_names():
    """Return the 15 class names (capitalized, matching dataset folder names)."""
    return [
        "Beetle", "Butterfly", "Cat", "Cow", "Dog",
        "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
        "Mouse", "Panda", "Spider", "Tiger", "Zebra",
    ]


@pytest.fixture(scope="session")
def num_classes():
    return 15


@pytest.fixture(scope="session")
def image_size():
    return (224, 224)


@pytest.fixture(scope="session")
def dummy_image_array(image_size):
    """Return a random float32 image array (1, 224, 224, 3) in [0, 1]."""
    np.random.seed(42)
    return np.random.rand(1, image_size[0], image_size[1], 3).astype(np.float32)


@pytest.fixture(scope="session")
def dummy_batch(image_size):
    """Return a batch of 4 random images."""
    np.random.seed(42)
    return np.random.rand(4, image_size[0], image_size[1], 3).astype(np.float32)


@pytest.fixture(scope="session")
def built_model(num_classes):
    """Build and return a compiled VGG-16 model (frozen base)."""
    from src.models.vgg16_model import build_vgg16_model, compile_model
    model = build_vgg16_model(num_classes=num_classes, freeze_base=True)
    model = compile_model(model, learning_rate=0.0001)
    return model


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory and clean it up after the test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="function")
def fake_dataset_dir(temp_dir, class_names):
    """
    Create a minimal fake dataset directory with 5 dummy images per class.
    Structure: temp_dir/raw/animal-image-classification/<class>/img_NNN.jpg
    """
    raw_dir = temp_dir / "raw" / "animal-image-classification"
    for cls in class_names[:3]:  # Only 3 classes for speed
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(5):
            img = Image.new("RGB", (300, 300), color=(i * 40, i * 20, i * 10))
            img.save(cls_dir / f"img_{i:03d}.jpg")
    return raw_dir


@pytest.fixture(scope="session")
def flask_app():
    """Create a Flask test client (model not loaded — tests health endpoint only)."""
    from app.app import create_app
    app = create_app(model_path="nonexistent_model.h5")
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="session")
def flask_client(flask_app):
    return flask_app.test_client()
