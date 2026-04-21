"""
test_app.py
-----------
Unit tests for the Flask web application.
"""

import sys
import io
import json
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_image_bytes(size=(224, 224), color=(100, 150, 200)) -> bytes:
    """Create a dummy JPEG image as bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, flask_client):
        resp = flask_client.get("/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, flask_client):
        resp = flask_client.get("/health")
        data = resp.get_json()
        assert "status" in data
        assert "model_loaded" in data
        assert "classes" in data
        assert "num_classes" in data

    def test_health_has_15_classes(self, flask_client):
        resp = flask_client.get("/health")
        data = resp.get_json()
        assert data["num_classes"] == 15
        assert len(data["classes"]) == 15

    def test_health_class_names(self, flask_client):
        resp = flask_client.get("/health")
        data = resp.get_json()
        expected = {
            "Beetle", "Butterfly", "Cat", "Cow", "Dog",
            "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
            "Mouse", "Panda", "Spider", "Tiger", "Zebra",
        }
        assert set(data["classes"]) == expected


class TestIndexEndpoint:
    """Tests for GET /."""

    def test_index_returns_200(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self, flask_client):
        resp = flask_client.get("/")
        assert b"Animal Classifier" in resp.data
        assert b"text/html" in resp.content_type.encode()


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_no_file_returns_400(self, flask_client):
        resp = flask_client.post("/predict")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_empty_filename_returns_400(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_predict_unsupported_format_returns_400(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data={"file": (io.BytesIO(b"fake"), "file.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_without_model_returns_503(self, flask_client):
        """When model is not loaded, should return 503."""
        img_bytes = make_image_bytes()
        resp = flask_client.post(
            "/predict",
            data={"file": (io.BytesIO(img_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )
        # Model not loaded → 503
        assert resp.status_code == 503

    def test_predict_corrupt_image_returns_400(self, flask_client):
        resp = flask_client.post(
            "/predict",
            data={"file": (io.BytesIO(b"not_an_image"), "bad.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


class TestPreprocessImage:
    """Tests for the image preprocessing function."""

    def test_output_shape(self):
        from app.app import preprocess_image
        img_bytes = make_image_bytes(size=(500, 300))
        arr = preprocess_image(img_bytes)
        assert arr.shape == (1, 224, 224, 3)

    def test_pixel_range(self):
        from app.app import preprocess_image
        img_bytes = make_image_bytes()
        arr = preprocess_image(img_bytes)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_dtype_float32(self):
        from app.app import preprocess_image
        img_bytes = make_image_bytes()
        arr = preprocess_image(img_bytes)
        assert arr.dtype == np.float32

    def test_grayscale_converted_to_rgb(self):
        from app.app import preprocess_image
        gray = Image.new("L", (100, 100), color=128)
        buf = io.BytesIO()
        gray.save(buf, format="JPEG")
        arr = preprocess_image(buf.getvalue())
        assert arr.shape[-1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
