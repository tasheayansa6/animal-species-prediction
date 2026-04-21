"""
predict.py
----------
Make predictions on new images using the trained model.

Usage:
    # Single image
    python scripts/predict.py --image path/to/image.jpg

    # Directory of images
    python scripts/predict.py --image-dir path/to/images/

    # Save results to CSV
    python scripts/predict.py --image-dir path/to/images/ --output predictions.csv

    # Show top-3 predictions
    python scripts/predict.py --image path/to/image.jpg --top-k 3
"""

import sys
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.model_utils import load_model
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger

setup_root_logger()
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Parameters
    ----------
    image_path : str
    target_size : tuple

    Returns
    -------
    np.ndarray of shape (1, H, W, 3), values in [0, 1]
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(
    model: tf.keras.Model,
    image_path: str,
    class_names: List[str],
    top_k: int = 1,
    image_size: Tuple[int, int] = (224, 224),
) -> List[Dict]:
    """
    Predict the class of a single image.

    Parameters
    ----------
    model : tf.keras.Model
    image_path : str
    class_names : list of str
    top_k : int
        Number of top predictions to return.
    image_size : tuple

    Returns
    -------
    List of dicts with keys: rank, class, confidence
    """
    img_array = preprocess_image(image_path, target_size=image_size)
    proba = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(proba)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank":       rank,
            "class":      class_names[idx],
            "confidence": float(proba[idx]),
        })
    return results


def predict_directory(
    model: tf.keras.Model,
    image_dir: str,
    class_names: List[str],
    top_k: int = 1,
    image_size: Tuple[int, int] = (224, 224),
) -> List[Dict]:
    """
    Predict classes for all images in a directory.

    Returns
    -------
    List of dicts with keys: image_path, rank, class, confidence
    """
    image_dir = Path(image_dir)
    image_paths = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        logger.warning("No images found in %s", image_dir)
        return []

    logger.info("Found %d images in %s", len(image_paths), image_dir)
    all_results = []

    for img_path in image_paths:
        try:
            preds = predict_single(model, str(img_path), class_names, top_k, image_size)
            for pred in preds:
                all_results.append({"image_path": str(img_path), **pred})
        except Exception as e:
            logger.warning("Could not process %s: %s", img_path, e)

    return all_results


def save_predictions_csv(predictions: List[Dict], output_path: str) -> None:
    """Save predictions to a CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not predictions:
        logger.warning("No predictions to save.")
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    logger.info("Predictions saved to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict animal species from images")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Path to model file")
    parser.add_argument("--image", default=None, help="Path to a single image")
    parser.add_argument("--image-dir", default=None, help="Path to a directory of images")
    parser.add_argument(
        "--output",
        default="outputs/predictions/predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    config = load_config(args.config)
    image_size = tuple(config["data"]["image_size"])
    class_names = config["data"]["classes"]

    model_path = args.model or os.path.join(
        config["paths"]["final_model_dir"], "animal_classifier.h5"
    )
    logger.info("Loading model: %s", model_path)
    model = load_model(model_path)

    if args.image:
        preds = predict_single(model, args.image, class_names, args.top_k, image_size)
        print(f"\nPredictions for: {args.image}")
        print("-" * 40)
        for p in preds:
            print(f"  #{p['rank']}  {p['class']:<15}  {p['confidence']*100:.2f}%")

    elif args.image_dir:
        all_preds = predict_directory(model, args.image_dir, class_names, args.top_k, image_size)
        save_predictions_csv(all_preds, args.output)

        # Print summary
        if all_preds:
            top1 = [p for p in all_preds if p["rank"] == 1]
            from collections import Counter
            class_counts = Counter(p["class"] for p in top1)
            print("\nPrediction Summary:")
            for cls, count in class_counts.most_common():
                print(f"  {cls:<15} {count} images")


if __name__ == "__main__":
    main()
