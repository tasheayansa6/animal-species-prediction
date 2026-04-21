"""
callbacks.py
------------
Custom and standard Keras callbacks for training.
"""

import os
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    phase: str = "phase1",
    patience: int = 7,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> List[tf.keras.callbacks.Callback]:
    """
    Build and return a list of Keras callbacks.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save model checkpoints.
    log_dir : str
        Directory for TensorBoard logs and CSV logs.
    phase : str
        Training phase label ('phase1' or 'phase2').
    patience : int
        EarlyStopping patience (epochs without improvement).
    reduce_lr_patience : int
        ReduceLROnPlateau patience.
    reduce_lr_factor : float
        Factor by which to reduce LR.
    min_lr : float
        Minimum learning rate.

    Returns
    -------
    List of tf.keras.callbacks.Callback
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = []

    # 1. Save best model checkpoint
    best_ckpt_path = os.path.join(checkpoint_dir, f"{phase}_best.h5")
    callbacks.append(
        ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        )
    )
    logger.info("ModelCheckpoint → %s", best_ckpt_path)

    # 2. Save weights every epoch (lightweight — just weights, not full model)
    epoch_ckpt_path = os.path.join(checkpoint_dir, f"{phase}_epoch_{{epoch:02d}}.weights.h5")
    callbacks.append(
        ModelCheckpoint(
            filepath=epoch_ckpt_path,
            monitor="val_accuracy",
            save_best_only=False,
            save_weights_only=True,   # weights only = much smaller file
            mode="max",
            verbose=0,
        )
    )

    # 3. Early stopping
    callbacks.append(
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        )
    )

    # 4. Reduce learning rate on plateau
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        )
    )

    # 5. TensorBoard
    tb_log_dir = os.path.join(log_dir, "tensorboard", phase)
    callbacks.append(
        TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        )
    )
    logger.info("TensorBoard logs → %s", tb_log_dir)

    # 6. CSV Logger
    csv_path = os.path.join(log_dir, f"{phase}_log.csv")
    callbacks.append(CSVLogger(csv_path, append=False))
    logger.info("CSVLogger → %s", csv_path)

    return callbacks


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Logs the current learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logger.info("Epoch %d — Learning rate: %.8f", epoch + 1, lr)
        if logs is not None:
            logs["lr"] = lr
