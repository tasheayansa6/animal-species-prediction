"""
train_safe.py
-------------
Memory-safe training script designed for low-RAM machines (< 4 GB free).

Key differences from train.py:
  - Uses tf.data pipeline with prefetch(1) — only 1 batch in RAM at a time
  - Saves a full model checkpoint after EVERY epoch
  - Resumes from the latest checkpoint if training was interrupted
  - Trains one epoch at a time in a loop so Python can garbage-collect between epochs
  - Disables augmentation by default (saves memory)

Usage:
    python scripts/train_safe.py
    python scripts/train_safe.py --resume          # resume from last checkpoint
    python scripts/train_safe.py --epochs 5        # train only 5 epochs
"""

import sys
import os
import gc
import argparse
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_root_logger
from src.utils.visualization import plot_training_history

setup_root_logger()
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]
IMG_SIZE   = (128, 128)
NUM_CLASSES = 15


# ── tf.data pipeline (memory-efficient) ──────────────────────────────

def make_dataset(directory: str, batch_size: int, training: bool) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a directory of class subfolders.
    Uses prefetch(1) so only one batch is in RAM at a time.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=training,
        seed=42,
    )
    # Normalize to [0, 1]
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=1,
    )
    # prefetch(1) = only 1 batch pre-loaded — minimal RAM usage
    ds = ds.prefetch(1)
    return ds


# ── Build model ───────────────────────────────────────────────────────

def build_model(input_shape=(128, 128, 3)) -> tf.keras.Model:
    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="VGG16_AnimalClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model


# ── Find latest checkpoint ────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: str):
    """Return (epoch_number, filepath) of the most recent epoch checkpoint, or (0, None)."""
    ckpt_dir = Path(ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("safe_epoch_*.h5"))
    if not checkpoints:
        return 0, None
    latest = checkpoints[-1]
    # filename: safe_epoch_03.h5 → epoch 3
    try:
        epoch_num = int(latest.stem.split("_")[-1])
    except ValueError:
        epoch_num = 0
    return epoch_num, str(latest)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Memory-safe VGG-16 training")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--epochs",   type=int, default=10)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    config     = load_config(args.config)
    ckpt_dir   = config["paths"]["checkpoints_dir"]
    final_dir  = config["paths"]["final_model_dir"]
    figures_dir = config["paths"]["figures_dir"]
    logs_dir   = config["paths"]["logs_dir"]
    train_dir  = config["paths"]["train_dir"]
    val_dir    = config["paths"]["val_dir"]

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    batch_size  = args.batch
    total_epochs = args.epochs

    logger.info("=" * 55)
    logger.info("Memory-Safe VGG-16 Training")
    logger.info("  image_size : %s", IMG_SIZE)
    logger.info("  batch_size : %d", batch_size)
    logger.info("  epochs     : %d", total_epochs)
    logger.info("  train_dir  : %s", train_dir)
    logger.info("=" * 55)

    # ── Build datasets ────────────────────────────────────────────────
    logger.info("Building tf.data pipelines...")
    train_ds = make_dataset(train_dir, batch_size, training=True)
    val_ds   = make_dataset(val_dir,   batch_size, training=False)

    n_train = sum(1 for _ in train_ds) * batch_size
    n_val   = sum(1 for _ in val_ds)   * batch_size
    logger.info("Train batches: %d (~%d images)", len(train_ds), n_train)
    logger.info("Val   batches: %d (~%d images)", len(val_ds),   n_val)

    # ── Build or load model ───────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch, ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path:
            logger.info("Resuming from epoch %d: %s", start_epoch, ckpt_path)
            model = tf.keras.models.load_model(ckpt_path)
        else:
            logger.info("No checkpoint found — starting fresh.")
            model = build_model()
    else:
        model = build_model()

    total_params    = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    logger.info("Params — total: %s  trainable: %s  frozen: %s",
                f"{total_params:,}", f"{trainable_params:,}",
                f"{total_params - trainable_params:,}")

    # ── Training loop — one epoch at a time ──────────────────────────
    history_log = []
    best_val_acc = 0.0
    patience_counter = 0
    patience = 4

    log_path = Path(logs_dir) / "safe_training_log.csv"

    for epoch in range(start_epoch, total_epochs):
        logger.info("─" * 55)
        logger.info("Epoch %d / %d", epoch + 1, total_epochs)
        t0 = time.time()

        hist = model.fit(
            train_ds,
            epochs=1,
            validation_data=val_ds,
            verbose=1,
        )

        elapsed = (time.time() - t0) / 60
        train_acc = hist.history["accuracy"][0]
        val_acc   = hist.history["val_accuracy"][0]
        train_loss = hist.history["loss"][0]
        val_loss   = hist.history["val_loss"][0]

        logger.info(
            "Epoch %d done in %.1f min — train_acc=%.4f  val_acc=%.4f",
            epoch + 1, elapsed, train_acc, val_acc,
        )

        # Save epoch checkpoint (full model — can resume from it)
        ckpt_path = str(Path(ckpt_dir) / f"safe_epoch_{epoch+1:02d}.h5")
        model.save(ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = str(Path(ckpt_dir) / "safe_best.h5")
            model.save(best_path)
            logger.info("New best val_acc=%.4f — saved: %s", best_val_acc, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                "No improvement (best=%.4f). Patience %d/%d",
                best_val_acc, patience_counter, patience,
            )

        # Log to CSV
        row = {
            "epoch": epoch + 1,
            "train_acc": round(train_acc, 4),
            "val_acc":   round(val_acc, 4),
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "elapsed_min": round(elapsed, 1),
        }
        history_log.append(row)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Force garbage collection between epochs
        gc.collect()

        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered after %d epochs.", epoch + 1)
            break

    # ── Save final model ──────────────────────────────────────────────
    final_path = str(Path(final_dir) / "animal_classifier.h5")
    # Load best weights before saving final
    best_path = str(Path(ckpt_dir) / "safe_best.h5")
    if Path(best_path).exists():
        model = tf.keras.models.load_model(best_path)
        logger.info("Loaded best model from: %s", best_path)
    model.save(final_path)
    logger.info("Final model saved: %s", final_path)

    # ── Plot training history ─────────────────────────────────────────
    if history_log:
        history_dict = {
            "accuracy":     [r["train_acc"]  for r in history_log],
            "val_accuracy": [r["val_acc"]    for r in history_log],
            "loss":         [r["train_loss"] for r in history_log],
            "val_loss":     [r["val_loss"]   for r in history_log],
        }
        plot_training_history(
            history_dict,
            save_path=f"{figures_dir}/training_history.png",
            title=f"Training History ({len(history_log)} epochs)",
        )

    logger.info("=" * 55)
    logger.info("Training complete!")
    logger.info("  Best val_accuracy : %.4f (%.2f%%)", best_val_acc, best_val_acc * 100)
    logger.info("  Final model       : %s", final_path)
    logger.info("  Training log      : %s", log_path)
    logger.info("=" * 55)
    logger.info("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
