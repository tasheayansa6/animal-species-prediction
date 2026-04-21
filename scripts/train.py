"""
train.py
--------
Main training script for the animal species classifier.
Optimized for both CPU and GPU training.

Usage:
    python scripts/train.py
    python scripts/train.py --skip-fine-tuning
    python scripts/train.py --no-class-weights
    python scripts/train.py --no-augmentation
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── TensorFlow memory & thread config (must be before tf import) ──────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress verbose TF logs
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf

# Limit GPU memory growth (no-op on CPU, safe to always set)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from src.training.trainer import Trainer
from src.utils.logger import get_logger, setup_root_logger
from src.utils.visualization import (
    plot_training_history,
    merge_histories,
)
from src.utils.config import load_config

setup_root_logger()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the animal species classifier")
    parser.add_argument("--config",            default="config/config.yaml")
    parser.add_argument("--skip-fine-tuning",  action="store_true")
    parser.add_argument("--no-class-weights",  action="store_true")
    parser.add_argument("--no-augmentation",   action="store_true")
    args = parser.parse_args()

    if gpus:
        logger.info("GPUs: %s", [g.name for g in gpus])
    else:
        logger.warning(
            "No GPU detected — training on CPU.\n"
            "  Phase 1 (~20 epochs × 30k images) will take several hours.\n"
            "  Tip: reduce batch_size in config.yaml if you get memory errors."
        )

    config = load_config(args.config)
    logger.info(
        "Config: batch_size=%d  epochs=%d  image_size=%s",
        config["data"]["batch_size"],
        config["training"]["epochs"],
        config["data"]["image_size"],
    )

    trainer = Trainer(config_path=args.config)
    result = trainer.train(
        use_class_weights=not args.no_class_weights,
        augment_train=not args.no_augmentation,
        skip_fine_tuning=args.skip_fine_tuning,
    )

    # ── Save training plots ───────────────────────────────────────────
    figures_dir = config["paths"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

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

    logger.info("Done. Model saved to: %s", config["paths"]["final_model_dir"])
    logger.info("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
