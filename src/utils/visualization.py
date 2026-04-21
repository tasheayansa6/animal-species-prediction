"""
visualization.py
----------------
Plotting utilities for training history, confusion matrix,
sample predictions, and class distribution.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]


def plot_training_history(
    history: Dict,
    save_path: str = "outputs/figures/training_history.png",
    title: str = "Training History",
) -> None:
    """
    Plot training and validation accuracy/loss curves.

    Parameters
    ----------
    history : dict
        Keras history.history dict (or merged phase1 + phase2).
    save_path : str
    title : str
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    # Accuracy
    axes[0].plot(epochs, history.get("accuracy", []), "b-o", label="Train Accuracy", markersize=4)
    axes[0].plot(epochs, history.get("val_accuracy", []), "r-o", label="Val Accuracy", markersize=4)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history.get("loss", []), "b-o", label="Train Loss", markersize=4)
    axes[1].plot(epochs, history.get("val_loss", []), "r-o", label="Val Loss", markersize=4)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training history plot saved: %s", save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "outputs/figures/confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (from sklearn.metrics.confusion_matrix).
    class_names : list of str, optional
    save_path : str
    normalize : bool
        If True, normalize by row (true label counts).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved: %s", save_path)


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "outputs/figures/sample_predictions.png",
    n_samples: int = 16,
) -> None:
    """
    Plot a grid of sample images with true and predicted labels.

    Parameters
    ----------
    images : np.ndarray
        Array of images, shape (N, H, W, C), values in [0, 1].
    y_true : np.ndarray
        True class indices.
    y_pred : np.ndarray
        Predicted class indices.
    class_names : list of str, optional
    save_path : str
    n_samples : int
        Number of samples to display (must be a perfect square or ≤ 16).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"
        ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Sample predictions plot saved: %s", save_path)


def plot_class_distribution(
    generator: object,
    class_names: Optional[List[str]] = None,
    save_path: str = "outputs/figures/class_distribution.png",
    title: str = "Class Distribution",
) -> None:
    """
    Plot the class distribution of a data generator.

    Parameters
    ----------
    generator : DirectoryIterator
        Keras data generator with .classes attribute.
    class_names : list of str, optional
    save_path : str
    title : str
    """
    if class_names is None:
        class_names = CLASS_NAMES

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    classes = generator.classes
    counts = np.bincount(classes, minlength=len(class_names))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(class_names, counts, color=sns.color_palette("husl", len(class_names)))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Class distribution plot saved: %s", save_path)


def merge_histories(history1: Dict, history2: Dict) -> Dict:
    """
    Concatenate two Keras history dicts (Phase 1 + Phase 2).

    Parameters
    ----------
    history1, history2 : dict

    Returns
    -------
    Merged dict with concatenated metric lists.
    """
    merged = {}
    for key in history1:
        if key in history2:
            merged[key] = list(history1[key]) + list(history2[key])
        else:
            merged[key] = list(history1[key])
    for key in history2:
        if key not in merged:
            merged[key] = list(history2[key])
    return merged
