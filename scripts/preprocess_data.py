"""
preprocess_data.py
------------------
Runs the full preprocessing pipeline:
  - Reads raw images from data/raw/animal-image-classification
  - Splits into train / validation / test (70/15/15)
  - Resizes all images to 224×224
  - Saves to data/processed/
  - Resizes all images to 224×224
  - Saves to data/processed/

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --force   # Re-process even if already done
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import DataPreprocessor
from src.utils.logger import get_logger, setup_root_logger

setup_root_logger()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Animal Image Classification dataset (15 classes)")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if processed data already exists",
    )
    args = parser.parse_args()

    logger.info("Starting data preprocessing...")
    preprocessor = DataPreprocessor(config_path=args.config)
    preprocessor.run(force=args.force)

    # Print statistics
    stats = preprocessor.get_split_statistics()
    logger.info("\nDataset Statistics:")
    logger.info("%-15s %-10s %-10s %-10s", "Class", "Train", "Val", "Test")
    logger.info("-" * 50)

    all_classes = set()
    for split_stats in stats.values():
        all_classes.update(split_stats.keys())

    for cls in sorted(all_classes):
        train_n = stats.get("train", {}).get(cls, 0)
        val_n   = stats.get("validation", {}).get(cls, 0)
        test_n  = stats.get("test", {}).get(cls, 0)
        logger.info("%-15s %-10d %-10d %-10d", cls, train_n, val_n, test_n)

    totals = {
        split: sum(counts.values())
        for split, counts in stats.items()
    }
    logger.info("-" * 50)
    logger.info(
        "%-15s %-10d %-10d %-10d",
        "TOTAL",
        totals.get("train", 0),
        totals.get("validation", 0),
        totals.get("test", 0),
    )
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
