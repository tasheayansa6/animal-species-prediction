"""
trainer.py
----------
Main training loop for the animal species classifier.

Two-phase transfer learning:
  Phase 1 — Feature Extraction (VGG-16 frozen, train head only)
  Phase 2 — Fine-Tuning (unfreeze last N VGG-16 layers)

CPU-optimised: uses tf.data pipeline with prefetch + cache for
validation, and logs epoch timing so you can estimate total time.
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from src.data.dataset import AnimalDataset
from src.models.transfer_learning import TransferLearningManager
from src.models.model_utils import save_model
from src.training.callbacks import build_callbacks
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class Trainer:
    """
    Orchestrates the full two-phase training pipeline.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config    = load_config(config_path)
        self.train_cfg = self.config["training"]
        self.paths     = self.config["paths"]

        self.dataset    = AnimalDataset(config_path)
        self.tl_manager = TransferLearningManager(config_path)

        self.history_phase1 = None
        self.history_phase2 = None
        self.model: Optional[Model] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        use_class_weights: bool = True,
        augment_train: bool = True,
        skip_fine_tuning: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full two-phase training pipeline.

        Parameters
        ----------
        use_class_weights : bool
            Apply class weights to handle imbalance.
        augment_train : bool
            Apply data augmentation to the training set.
        skip_fine_tuning : bool
            If True, only run Phase 1.

        Returns
        -------
        dict with 'phase1_history' and optionally 'phase2_history'.
        """
        # ── Build generators ─────────────────────────────────────────
        train_gen, val_gen, test_gen = self.dataset.build_generators(
            augment_train=augment_train
        )

        logger.info(
            "Dataset: train=%d  val=%d  test=%d  batch=%d  steps/epoch=%d",
            train_gen.samples,
            val_gen.samples,
            test_gen.samples,
            train_gen.batch_size,
            len(train_gen),
        )

        class_weights = self.dataset.get_class_weights() if use_class_weights else None

        # ── Phase 1: Feature Extraction ──────────────────────────────
        logger.info("=" * 60)
        logger.info("PHASE 1 — Feature Extraction (VGG-16 frozen)")
        logger.info("=" * 60)

        self.model = self.tl_manager.build_feature_extraction_model()
        self._log_param_counts()

        callbacks_p1 = build_callbacks(
            checkpoint_dir=self.paths["checkpoints_dir"],
            log_dir=self.paths["logs_dir"],
            phase="phase1",
            patience=self.train_cfg["early_stopping_patience"],
            reduce_lr_patience=self.train_cfg["reduce_lr_patience"],
            reduce_lr_factor=self.train_cfg["reduce_lr_factor"],
            min_lr=self.train_cfg["min_lr"],
        )

        t0 = time.time()
        self.history_phase1 = self.model.fit(
            train_gen,
            epochs=self.train_cfg["epochs"],
            validation_data=val_gen,
            callbacks=callbacks_p1,
            class_weight=class_weights,
            verbose=1,
        )
        elapsed = (time.time() - t0) / 60
        logger.info("Phase 1 done in %.1f min.", elapsed)

        # Restore best Phase 1 weights
        best_p1 = os.path.join(self.paths["checkpoints_dir"], "phase1_best.h5")
        if os.path.exists(best_p1):
            self.model.load_weights(best_p1)
            logger.info("Restored best Phase 1 weights.")

        result = {"phase1_history": self.history_phase1.history}

        # ── Phase 2: Fine-Tuning ──────────────────────────────────────
        if not skip_fine_tuning:
            logger.info("=" * 60)
            logger.info("PHASE 2 — Fine-Tuning (last VGG-16 blocks unfrozen)")
            logger.info("=" * 60)

            self.model = self.tl_manager.prepare_fine_tuning(self.model)
            self._log_param_counts()

            callbacks_p2 = build_callbacks(
                checkpoint_dir=self.paths["checkpoints_dir"],
                log_dir=self.paths["logs_dir"],
                phase="phase2",
                patience=self.train_cfg["early_stopping_patience"],
                reduce_lr_patience=self.train_cfg["reduce_lr_patience"],
                reduce_lr_factor=self.train_cfg["reduce_lr_factor"],
                min_lr=self.train_cfg["min_lr"],
            )

            fine_tune_epochs = self.config["transfer_learning"].get("fine_tune_epochs", 5)
            t0 = time.time()
            self.history_phase2 = self.model.fit(
                train_gen,
                epochs=fine_tune_epochs,
                validation_data=val_gen,
                callbacks=callbacks_p2,
                class_weight=class_weights,
                verbose=1,
            )
            elapsed = (time.time() - t0) / 60
            logger.info("Phase 2 done in %.1f min.", elapsed)

            best_p2 = os.path.join(self.paths["checkpoints_dir"], "phase2_best.h5")
            if os.path.exists(best_p2):
                self.model.load_weights(best_p2)
                logger.info("Restored best Phase 2 weights.")

            result["phase2_history"] = self.history_phase2.history

        # ── Save ─────────────────────────────────────────────────────
        self._save_final_model()
        self._save_training_log(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_param_counts(self) -> None:
        total     = self.model.count_params()
        trainable = int(sum(tf.size(w).numpy() for w in self.model.trainable_weights))
        logger.info(
            "Parameters — total: %s  trainable: %s  frozen: %s",
            f"{total:,}", f"{trainable:,}", f"{total - trainable:,}",
        )

    def _save_final_model(self) -> None:
        final_dir = Path(self.paths["final_model_dir"])
        final_dir.mkdir(parents=True, exist_ok=True)

        h5_path = str(final_dir / "animal_classifier.h5")
        save_model(self.model, h5_path, save_format="h5")
        logger.info("Final model (H5) saved: %s", h5_path)

        try:
            saved_model_path = str(final_dir / "saved_model")
            save_model(self.model, saved_model_path, save_format="tf")
            logger.info("Final model (SavedModel) saved: %s", saved_model_path)
        except Exception as e:
            logger.warning("SavedModel export failed (non-critical): %s", e)

    def _save_training_log(self, result: Dict[str, Any]) -> None:
        log_dir  = Path(self.paths["logs_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "training_log.csv"

        rows = []
        for phase_key, history in result.items():
            phase_name = phase_key.replace("_history", "")
            accs = history.get("accuracy", [])
            for i, acc in enumerate(accs):
                rows.append({
                    "phase":        phase_name,
                    "epoch":        i + 1,
                    "accuracy":     round(float(acc), 4),
                    "val_accuracy": round(float(history.get("val_accuracy", [0])[i]), 4),
                    "loss":         round(float(history.get("loss", [0])[i]), 4),
                    "val_loss":     round(float(history.get("val_loss", [0])[i]), 4),
                })

        if rows:
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Training log: %s", log_path)
