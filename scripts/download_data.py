"""
download_data.py
----------------
Downloads the Animal Image Classification Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset
Classes: bear, bird, cat, cow, deer, dog, dolphin, elephant,
         giraffe, horse, kangaroo, lion, panda, tiger, zebra

Usage:
    python scripts/download_data.py

Prerequisites:
    1. Install the Kaggle API:  pip install kaggle
    2. Place your kaggle.json credentials in ~/.kaggle/kaggle.json
       (download from kaggle.com → Account → API → Create New Token)
    3. Accept the dataset terms on Kaggle if required.
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET = "utkarshsaxenadn/animal-image-classification-dataset"

EXPECTED_CLASSES = [
    "bear", "bird", "cat", "cow", "deer",
    "dog", "dolphin", "elephant", "giraffe", "horse",
    "kangaroo", "lion", "panda", "tiger", "zebra",
]


def download_dataset(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_data"])

    # Check if already downloaded
    if raw_dir.exists():
        existing = [d.name.lower() for d in raw_dir.iterdir() if d.is_dir()]
        if any(cls in existing for cls in EXPECTED_CLASSES):
            logger.info("Raw data already exists at %s. Skipping download.", raw_dir)
            return

    raw_dir.mkdir(parents=True, exist_ok=True)
    download_dir = raw_dir.parent  # data/raw/

    logger.info("Downloading Animal Image Classification Dataset from Kaggle...")
    logger.info("Dataset: %s", KAGGLE_DATASET)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(download_dir),
            unzip=True,
        )
        logger.info("Download and extraction complete.")
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        _print_manual_instructions(raw_dir)
        sys.exit(1)
    except Exception as e:
        logger.error("Download failed: %s", e)
        _print_manual_instructions(raw_dir)
        sys.exit(1)

    # Locate and move extracted folders into raw_dir
    _organize_extracted_data(download_dir, raw_dir)

    # Verify
    found = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    logger.info("Found %d class folders: %s", len(found), sorted(found))
    missing = [c for c in EXPECTED_CLASSES if c not in [f.lower() for f in found]]
    if missing:
        logger.warning("Missing expected classes: %s", missing)
    else:
        logger.info("All 15 classes found.")


def _organize_extracted_data(download_dir: Path, raw_dir: Path) -> None:
    """
    Move extracted class folders into raw_dir.
    Handles various extraction structures:
      - flat:   download_dir/bear/, download_dir/cat/, ...
      - nested: download_dir/Animal Image Classification/Train/bear/, ...
    """
    # Search for class folders up to 3 levels deep
    for depth in range(3):
        for folder in _walk_dirs(download_dir, depth):
            if folder.name.lower() in EXPECTED_CLASSES:
                parent = folder.parent
                # Move all sibling class folders to raw_dir
                logger.info("Found class folders in: %s", parent)
                for sibling in parent.iterdir():
                    if sibling.is_dir() and sibling.name.lower() in EXPECTED_CLASSES:
                        dest = raw_dir / sibling.name.lower()
                        if not dest.exists():
                            shutil.move(str(sibling), str(dest))
                            logger.info("  Moved %s → %s", sibling.name, dest)
                return


def _walk_dirs(base: Path, depth: int):
    """Yield all directories at exactly `depth` levels below base."""
    if depth == 0:
        yield from (d for d in base.iterdir() if d.is_dir())
    else:
        for d in base.iterdir():
            if d.is_dir():
                yield from _walk_dirs(d, depth - 1)


def _print_manual_instructions(raw_dir: Path) -> None:
    logger.info(
        "\nManual download instructions:\n"
        "  1. Go to https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset\n"
        "  2. Click Download\n"
        "  3. Extract the ZIP\n"
        "  4. Copy the class folders into: %s\n"
        "     Expected folders: bear, bird, cat, cow, deer, dog, dolphin,\n"
        "                       elephant, giraffe, horse, kangaroo, lion,\n"
        "                       panda, tiger, zebra",
        raw_dir,
    )


if __name__ == "__main__":
    download_dataset()
