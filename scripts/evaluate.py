"""
evaluate.py
-----------
Evaluate a trained model on the test set and generate reports.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/final/animal_classifier.h5
    python scripts/evaluate.py --split validation
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.data.dataset import AnimalDataset
from src.models.model_utils import load_model
from src.training.metrics import compute_metrics, format_metrics_report
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_sample_predictions,
)

setup_root_logger()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained animal classifier")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model file. Defaults to models/final/animal_classifier.h5",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation", "train"],
        default="test",
        help="Which data split to evaluate on",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine model path
    model_path = args.model or os.path.join(
        config["paths"]["final_model_dir"], "animal_classifier.h5"
    )

    # Load model
    logger.info("Loading model from: %s", model_path)
    model = load_model(model_path)
    model.summary()

    # Build data generators
    dataset = AnimalDataset(config_path=args.config)
    train_gen, val_gen, test_gen = dataset.build_generators(augment_train=False)

    generator_map = {
        "train":      train_gen,
        "validation": val_gen,
        "test":       test_gen,
    }
    generator = generator_map[args.split]
    logger.info("Evaluating on %s set (%d samples)...", args.split, generator.samples)

    # Compute metrics
    class_names = config["data"]["classes"]
    metrics = compute_metrics(model, generator, class_names=class_names)

    # Format and print report
    report_str = format_metrics_report(metrics, class_names=class_names)
    print(report_str)

    # Save report
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"evaluation_report_{args.split}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info("Evaluation report saved: %s", report_path)

    # Plots
    figures_dir = config["paths"]["figures_dir"]

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        save_path=f"{figures_dir}/confusion_matrix_{args.split}.png",
        normalize=True,
    )

    # Sample predictions — grab one batch
    generator.reset()
    images_batch, labels_batch = next(iter(generator))
    y_pred_batch = np.argmax(model.predict(images_batch, verbose=0), axis=1)
    y_true_batch = np.argmax(labels_batch, axis=1)

    plot_sample_predictions(
        images=images_batch,
        y_true=y_true_batch,
        y_pred=y_pred_batch,
        class_names=class_names,
        save_path=f"{figures_dir}/sample_predictions_{args.split}.png",
        n_samples=16,
    )

    logger.info(
        "Final %s accuracy: %.4f (%.2f%%)",
        args.split,
        metrics["accuracy"],
        metrics["accuracy"] * 100,
    )


if __name__ == "__main__":
    main()
