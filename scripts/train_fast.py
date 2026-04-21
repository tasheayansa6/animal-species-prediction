"""
train_fast.py
-------------
Fast training script optimised for CPU with limited RAM.

Uses a SUBSET of training data (configurable) so each epoch
completes in ~5-15 minutes instead of 1.5 hours.

Default: 200 images/class (3,000 total) → ~8 min/epoch
Full:    2000 images/class (30,000 total) → ~90 min/epoch

Usage:
    # Quick run — 200 images/class, 10 epochs (~80 min total)
    python scripts/train_fast.py

    # Full dataset — 2000 images/class (leave overnight)
    python scripts/train_fast.py --samples-per-class 2000

    # Resume from last checkpoint
    python scripts/train_fast.py --resume
"""

import sys, os, gc, csv, time, argparse, shutil, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "2"
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

# ── Paths ─────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("src/data/Training Data/Training Data")
VAL_DIR     = Path("src/data/Validation Data/Validation Data")
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

for d in [CKPT_DIR, FINAL_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Build subset directory ────────────────────────────────────────────

def build_subset(samples_per_class: int, seed: int = 42) -> Path:
    """
    Copy a random subset of training images to a temp folder.
    Returns the path to the subset directory.
    """
    subset_dir = Path("src/data/_subset_train")
    # Check if already built with same size
    marker = subset_dir / f".built_{samples_per_class}"
    if marker.exists():
        print(f"Subset already built ({samples_per_class}/class). Reusing.")
        return subset_dir

    # Clean and rebuild
    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    random.seed(seed)
    total = 0
    for cls in CLASS_NAMES:
        src_cls = TRAIN_DIR / cls
        dst_cls = subset_dir / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = list(src_cls.glob("*.jpg")) + list(src_cls.glob("*.jpeg")) + \
                 list(src_cls.glob("*.png"))
        selected = random.sample(images, min(samples_per_class, len(images)))
        for img in selected:
            shutil.copy2(img, dst_cls / img.name)
        total += len(selected)

    marker.touch()
    print(f"Subset built: {total} images ({samples_per_class}/class) → {subset_dir}")
    return subset_dir


# ── Model ─────────────────────────────────────────────────────────────

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
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model


# ── Generators ────────────────────────────────────────────────────────

def make_generators(train_dir: Path, batch_size: int):
    dg = ImageDataGenerator(rescale=1.0/255)
    tg = dg.flow_from_directory(
        str(train_dir), target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="categorical", classes=CLASS_NAMES, shuffle=True, seed=42,
    )
    vg = dg.flow_from_directory(
        str(VAL_DIR), target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="categorical", classes=CLASS_NAMES, shuffle=False,
    )
    return tg, vg


# ── Checkpoint helpers ────────────────────────────────────────────────

def find_latest(prefix="fast"):
    ckpts = sorted(CKPT_DIR.glob(f"{prefix}_epoch_*.h5"))
    if not ckpts:
        return 0, None
    latest = ckpts[-1]
    try:
        epoch = int(latest.stem.split("_")[-1])
    except ValueError:
        epoch = 0
    return epoch, str(latest)


def log_row(row: dict, prefix="fast"):
    path = LOGS_DIR / f"{prefix}_training_log.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",            type=int, default=10)
    parser.add_argument("--batch",             type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=200,
                        help="Images per class for training (default 200 = fast)")
    parser.add_argument("--resume",            action="store_true")
    args = parser.parse_args()

    spc = args.samples_per_class
    total_train = spc * NUM_CLASSES

    print(f"\n{'='*55}")
    print(f"  Fast VGG-16 Training (CPU-optimised)")
    print(f"  samples/class : {spc}  ({total_train} total)")
    print(f"  batch_size    : {args.batch}")
    print(f"  epochs        : {args.epochs}")
    print(f"  est. time/ep  : ~{int(total_train/args.batch * 0.055)} min")
    print(f"{'='*55}\n")

    # Build subset
    train_dir = build_subset(spc)

    # Load or build model
    start_epoch = 0
    if args.resume:
        start_epoch, ckpt_path = find_latest("fast")
        if ckpt_path:
            print(f"Resuming from epoch {start_epoch}: {ckpt_path}\n")
            model = tf.keras.models.load_model(ckpt_path)
        else:
            print("No checkpoint found — starting fresh.\n")
            model = build_model()
    else:
        model = build_model()

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"Params — total:{total:,}  trainable:{trainable:,}  frozen:{total-trainable:,}\n")

    train_gen, val_gen = make_generators(train_dir, args.batch)
    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Steps/ep: {len(train_gen)}\n")

    best_val_acc     = 0.0
    patience_counter = 0
    patience         = 4
    history_log      = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch+1}/{args.epochs}")
        print(f"{'─'*55}")
        t0 = time.time()

        hist = model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=1)

        elapsed    = (time.time() - t0) / 60
        train_acc  = hist.history["accuracy"][0]
        val_acc    = hist.history["val_accuracy"][0]
        train_loss = hist.history["loss"][0]
        val_loss   = hist.history["val_loss"][0]
        top3       = hist.history.get("top3", [0])[0]

        print(f"\n  ✓ Epoch {epoch+1} — {elapsed:.1f} min")
        print(f"    train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  top3={top3:.4f}")

        # Save epoch checkpoint
        ckpt = str(CKPT_DIR / f"fast_epoch_{epoch+1:02d}.h5")
        model.save(ckpt)
        print(f"    Checkpoint: {ckpt}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(str(CKPT_DIR / "fast_best.h5"))
            print(f"    ★ New best val_acc={best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    Patience {patience_counter}/{patience} (best={best_val_acc:.4f})")

        row = {
            "epoch": epoch+1, "train_acc": round(train_acc,4),
            "val_acc": round(val_acc,4), "top3": round(top3,4),
            "train_loss": round(train_loss,4), "val_loss": round(val_loss,4),
            "min": round(elapsed,1),
        }
        history_log.append(row)
        log_row(row, "fast")

        gc.collect()
        tf.keras.backend.clear_session()

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

        # Reload for next epoch (clears TF graph)
        if epoch + 1 < args.epochs:
            model = tf.keras.models.load_model(str(CKPT_DIR / f"fast_epoch_{epoch+1:02d}.h5"))
            train_gen, val_gen = make_generators(train_dir, args.batch)

    # Save final
    best = str(CKPT_DIR / "fast_best.h5")
    if Path(best).exists():
        model = tf.keras.models.load_model(best)
    model.save(str(FINAL_DIR / "animal_classifier.h5"))

    # Print summary
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val_accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"\n  Epoch-by-epoch results:")
    print(f"  {'Ep':>3}  {'TrainAcc':>9}  {'ValAcc':>8}  {'Top3':>6}  {'Min':>5}")
    print(f"  {'─'*40}")
    for r in history_log:
        print(f"  {r['epoch']:>3}  {r['train_acc']:>9.4f}  {r['val_acc']:>8.4f}  "
              f"{r['top3']:>6.4f}  {r['min']:>5.1f}")
    print(f"{'='*55}")
    print("\nNext steps:")
    print("  python scripts/evaluate.py")
    print("  python app/app.py  →  http://localhost:5000")


if __name__ == "__main__":
    main()
