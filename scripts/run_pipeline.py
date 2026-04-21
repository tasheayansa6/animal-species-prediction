"""
run_pipeline.py
---------------
Runs the complete end-to-end pipeline in one command:
  1. Preprocess raw data
  2. Train the model (Phase 1 + Phase 2)
  3. Evaluate on the test set
  4. Export the final model

Usage:
    # Full pipeline
    python scripts/run_pipeline.py

    # Skip steps you've already done
    python scripts/run_pipeline.py --skip-preprocess
    python scripts/run_pipeline.py --skip-preprocess --skip-train

    # Feature extraction only (no fine-tuning)
    python scripts/run_pipeline.py --skip-fine-tuning

    # Force re-preprocessing even if processed data exists
    python scripts/run_pipeline.py --force-preprocess
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.data.preprocessing import DataPreprocessor
from src.training.trainer import Trainer
from src.data.dataset import AnimalDataset
from src.models.model_utils import load_model
from src.training.metrics import compute_metrics, format_metrics_report
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_class_distribution,
    merge_histories,
)

setup_root_logger()
logger = get_logger(__name__)

import numpy as np


def step_preprocess(config: dict, force: bool = False) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1 — Data Preprocessing")
    logger.info("=" * 60)
    t0 = time.time()
    preprocessor = DataPreprocessor()
    preprocessor.run(force=force)
    stats = preprocessor.get_split_statistics()
    logger.info("Split statistics:")
    for split, counts in stats.items():
        total = sum(counts.values())
        logger.info("  %-12s %d images", split, total)
    logger.info("Preprocessing done in %.1fs", time.time() - t0)


def step_train(
    config: dict,
    skip_fine_tuning: bool = False,
    no_class_weights: bool = False,
    no_augmentation: bool = False,
) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2 — Training")
    logger.info("=" * 60)
    t0 = time.time()

    trainer = Trainer()
    result = trainer.train(
        use_class_weights=not no_class_weights,
        augment_train=not no_augmentation,
        skip_fine_tuning=skip_fine_tuning,
    )

    figures_dir = config["paths"]["figures_dir"]
    if "phase1_history" in result and "phase2_history" in result:
        merged = merge_histories(result["phase1_history"], result["phase2_history"])
        plot_training_history(
            merged,
            save_path=f"{figures_dir}/training_history.png",
            title="Training History (Phase 1 + Phase 2)",
        )
        plot_training_history(
            result["phase1_history"],
            save_path=f"{figures_dir}/training_history_phase1.png",
            title="Phase 1 — Feature Extraction",
        )
        plot_training_history(
            result["phase2_history"],
            save_path=f"{figures_dir}/training_history_phase2.png",
            title="Phase 2 — Fine-Tuning",
        )
    elif "phase1_history" in result:
        plot_training_history(
            result["phase1_history"],
            save_path=f"{figures_dir}/training_history.png",
            title="Training History",
        )

    logger.info("Training done in %.1f minutes.", (time.time() - t0) / 60)
    return result


def step_evaluate(config: dict) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3 — Evaluation")
    logger.info("=" * 60)

    model_path = str(
        Path(config["paths"]["final_model_dir"]) / "animal_classifier.h5"
    )
    if not Path(model_path).exists():
        logger.error("Model not found at %s. Run training first.", model_path)
        return

    model = load_model(model_path)
    dataset = AnimalDataset()
    train_gen, val_gen, test_gen = dataset.build_generators(augment_train=False)

    class_names = config["data"]["classes"]
    figures_dir = config["paths"]["figures_dir"]
    reports_dir = config["paths"]["reports_dir"]
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    for split_name, generator in [("validation", val_gen), ("test", test_gen)]:
        logger.info("Evaluating on %s set...", split_name)
        metrics = compute_metrics(model, generator, class_names=class_names)
        report_str = format_metrics_report(metrics, class_names=class_names)
        print(report_str)

        # Save text report
        report_path = Path(reports_dir) / f"evaluation_report_{split_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        logger.info("Report saved: %s", report_path)

        # Confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=class_names,
            save_path=f"{figures_dir}/confusion_matrix_{split_name}.png",
        )

        # Sample predictions
        generator.reset()
        imgs, lbls = next(iter(generator))
        y_pred = np.argmax(model.predict(imgs, verbose=0), axis=1)
        y_true = np.argmax(lbls, axis=1)
        plot_sample_predictions(
            imgs, y_true, y_pred,
            class_names=class_names,
            save_path=f"{figures_dir}/sample_predictions_{split_name}.png",
        )

    # Class distribution plot
    plot_class_distribution(
        train_gen,
        class_names=class_names,
        save_path=f"{figures_dir}/class_distribution_train.png",
        title="Training Set Class Distribution",
    )


def step_export(config: dict) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4 — Export")
    logger.info("=" * 60)

    model_path = str(
        Path(config["paths"]["final_model_dir"]) / "animal_classifier.h5"
    )
    if not Path(model_path).exists():
        logger.warning("Model not found — skipping export.")
        return

    model = load_model(model_path)
    final_dir = Path(config["paths"]["final_model_dir"])

    # TFLite export
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = final_dir / "animal_classifier.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        logger.info("TFLite model saved: %s (%.1f MB)", tflite_path, size_mb)
    except Exception as e:
        logger.warning("TFLite export failed: %s", e)

    # SavedModel export
    saved_model_dir = str(final_dir / "saved_model")
    model.save(saved_model_dir, save_format="tf")
    logger.info("SavedModel saved: %s", saved_model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full animal species classification pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-preprocess",   action="store_true")
    parser.add_argument("--skip-train",        action="store_true")
    parser.add_argument("--skip-evaluate",     action="store_true")
    parser.add_argument("--skip-export",       action="store_true")
    parser.add_argument("--skip-fine-tuning",  action="store_true")
    parser.add_argument("--force-preprocess",  action="store_true")
    parser.add_argument("--no-class-weights",  action="store_true")
    parser.add_argument("--no-augmentation",   action="store_true")
    args = parser.parse_args()

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPUs: %s", [g.name for g in gpus])
    else:
        logger.warning("No GPU — training on CPU.")

    config = load_config(args.config)
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("Animal Species Classification Pipeline")
    logger.info("Dataset: 15 classes | Model: VGG-16 Transfer Learning")
    logger.info("=" * 60)

    if not args.skip_preprocess:
        step_preprocess(config, force=args.force_preprocess)

    if not args.skip_train:
        step_train(
            config,
            skip_fine_tuning=args.skip_fine_tuning,
            no_class_weights=args.no_class_weights,
            no_augmentation=args.no_augmentation,
        )

    if not args.skip_evaluate:
        step_evaluate(config)

    if not args.skip_export:
        step_export(config)

    total_min = (time.time() - pipeline_start) / 60
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete in %.1f minutes.", total_min)
    logger.info("Outputs:")
    logger.info("  Models:  %s", config["paths"]["final_model_dir"])
    logger.info("  Figures: %s", config["paths"]["figures_dir"])
    logger.info("  Reports: %s", config["paths"]["reports_dir"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
