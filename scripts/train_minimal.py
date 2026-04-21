"""
train_minimal.py
----------------
Ultra-low-memory training script.
Works with as little as 2 GB free RAM.

Strategy:
  - Batch size 4 (tiny memory footprint)
  - Loads images one batch at a time via ImageDataGenerator (no caching)
  - Saves checkpoint after EVERY epoch
  - Resumes automatically from last saved epoch
  - Trains only the custom head (VGG-16 frozen) — Phase 1 only

Usage:
    python scripts/train_minimal.py
    python scripts/train_minimal.py --epochs 10
"""

import sys, os, gc, csv, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"
os.environ["OMP_NUM_THREADS"]           = "2"
os.environ["TF_NUM_INTRAOP_THREADS"]    = "2"
os.environ["TF_NUM_INTEROP_THREADS"]    = "1"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

# ── Config ────────────────────────────────────────────────────────────
TRAIN_DIR   = "src/data/Training Data/Training Data"
VAL_DIR     = "src/data/Validation Data/Validation Data"
TEST_DIR    = "src/data/Testing Data/Testing Data"
CKPT_DIR    = Path("models/checkpoints")
FINAL_DIR   = Path("models/final")
LOGS_DIR    = Path("models/logs")
FIGURES_DIR = Path("outputs/figures")

CLASS_NAMES = [
    "Beetle","Butterfly","Cat","Cow","Dog",
    "Elephant","Gorilla","Hippo","Lizard","Monkey",
    "Mouse","Panda","Spider","Tiger","Zebra",
]
IMG_SIZE    = (128, 128)
NUM_CLASSES = 15
LR          = 1e-4

for d in [CKPT_DIR, FINAL_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────

def find_latest_checkpoint():
    """Return (epoch_num, path) of latest safe_epoch_XX.h5, or (0, None)."""
    ckpts = sorted(CKPT_DIR.glob("minimal_epoch_*.h5"))
    if not ckpts:
        return 0, None
    latest = ckpts[-1]
    try:
        epoch = int(latest.stem.split("_")[-1])
    except ValueError:
        epoch = 0
    return epoch, str(latest)


def build_model():
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inp = Input(shape=(*IMG_SIZE, 3), name="input_image")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="VGG16_AnimalClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model


def make_generators(batch_size):
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="categorical", classes=CLASS_NAMES,
        shuffle=True, seed=42,
    )
    val_gen = datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="categorical", classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


def log_row(row: dict):
    log_path = LOGS_DIR / "minimal_training_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Ultra-Low-Memory VGG-16 Training")
    print(f"  image_size : {IMG_SIZE}")
    print(f"  batch_size : {args.batch}")
    print(f"  epochs     : {args.epochs}")
    print(f"{'='*55}\n")

    # ── Load or build model ───────────────────────────────────────────
    start_epoch, ckpt_path = find_latest_checkpoint()
    if ckpt_path:
        print(f"Resuming from epoch {start_epoch}: {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)
    else:
        print("Building new model...")
        model = build_model()

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"Params — total: {total:,}  trainable: {trainable:,}  frozen: {total-trainable:,}\n")

    # ── Generators ────────────────────────────────────────────────────
    train_gen, val_gen = make_generators(args.batch)
    print(f"Train: {train_gen.samples} images  |  Val: {val_gen.samples} images")
    print(f"Steps/epoch: {len(train_gen)}  |  Val steps: {len(val_gen)}\n")

    best_val_acc     = 0.0
    patience_counter = 0
    patience         = 4

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch+1}/{args.epochs}")
        print(f"{'─'*55}")
        t0 = time.time()

        hist = model.fit(
            train_gen,
            epochs=1,
            validation_data=val_gen,
            verbose=1,
        )

        elapsed    = (time.time() - t0) / 60
        train_acc  = hist.history["accuracy"][0]
        val_acc    = hist.history["val_accuracy"][0]
        train_loss = hist.history["loss"][0]
        val_loss   = hist.history["val_loss"][0]
        top3       = hist.history.get("top3", [0])[0]

        print(f"\nEpoch {epoch+1} done in {elapsed:.1f} min")
        print(f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  top3={top3:.4f}")

        # Save epoch checkpoint
        ckpt = str(CKPT_DIR / f"minimal_epoch_{epoch+1:02d}.h5")
        model.save(ckpt)
        print(f"  Checkpoint: {ckpt}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best = str(CKPT_DIR / "minimal_best.h5")
            model.save(best)
            print(f"  ★ New best val_acc={best_val_acc:.4f} → {best}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement (best={best_val_acc:.4f}). "
                  f"Patience {patience_counter}/{patience}")

        log_row({
            "epoch": epoch+1,
            "train_acc":  round(train_acc,  4),
            "val_acc":    round(val_acc,     4),
            "top3_acc":   round(top3,        4),
            "train_loss": round(train_loss,  4),
            "val_loss":   round(val_loss,    4),
            "min":        round(elapsed,     1),
        })

        # Free memory between epochs
        gc.collect()
        tf.keras.backend.clear_session()

        # Reload model for next epoch (clears TF graph memory)
        if epoch + 1 < args.epochs and patience_counter < patience:
            latest_ckpt = str(CKPT_DIR / f"minimal_epoch_{epoch+1:02d}.h5")
            model = tf.keras.models.load_model(latest_ckpt)
            train_gen, val_gen = make_generators(args.batch)

        if patience_counter >= patience:
            print(f"\nEarly stopping after epoch {epoch+1}.")
            break

    # ── Save final model ──────────────────────────────────────────────
    best = str(CKPT_DIR / "minimal_best.h5")
    if Path(best).exists():
        model = tf.keras.models.load_model(best)
    final = str(FINAL_DIR / "animal_classifier.h5")
    model.save(final)

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val_accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Final model       : {final}")
    print(f"  Log               : {LOGS_DIR}/minimal_training_log.csv")
    print(f"{'='*55}")
    print("\nNext: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
