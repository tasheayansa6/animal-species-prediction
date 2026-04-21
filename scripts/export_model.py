"""
export_model.py
---------------
Export the trained model in multiple formats for deployment.

Supported formats:
  - HDF5 (.h5)
  - TensorFlow SavedModel
  - TensorFlow Lite (.tflite)
  - ONNX (requires tf2onnx)

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --format tflite
    python scripts/export_model.py --format all
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.models.model_utils import load_model, save_model
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger

setup_root_logger()
logger = get_logger(__name__)


def export_tflite(model: tf.keras.Model, output_path: str) -> None:
    """Export model to TensorFlow Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("TFLite model saved: %s (%.1f MB)", output_path, size_mb)


def export_saved_model(model: tf.keras.Model, output_dir: str) -> None:
    """Export model in TensorFlow SavedModel format."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir, save_format="tf")
    logger.info("SavedModel exported: %s", output_dir)


def export_h5(model: tf.keras.Model, output_path: str) -> None:
    """Export model in HDF5 format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("HDF5 model saved: %s (%.1f MB)", output_path, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Source model path")
    parser.add_argument(
        "--format",
        choices=["h5", "savedmodel", "tflite", "all"],
        default="all",
        help="Export format",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    final_dir = Path(args.output_dir or config["paths"]["final_model_dir"])

    model_path = args.model or str(final_dir / "animal_classifier.h5")
    logger.info("Loading model: %s", model_path)
    model = load_model(model_path)

    fmt = args.format

    if fmt in ("h5", "all"):
        export_h5(model, str(final_dir / "animal_classifier.h5"))

    if fmt in ("savedmodel", "all"):
        export_saved_model(model, str(final_dir / "saved_model"))

    if fmt in ("tflite", "all"):
        export_tflite(model, str(final_dir / "animal_classifier.tflite"))

    logger.info("Export complete. Files in: %s", final_dir)


if __name__ == "__main__":
    main()
