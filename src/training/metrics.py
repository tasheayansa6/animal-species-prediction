"""
metrics.py
----------
Custom evaluation metrics and metric computation utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]


def compute_metrics(
    model: tf.keras.Model,
    generator: tf.keras.preprocessing.image.DirectoryIterator,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics on a data generator.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    generator : DirectoryIterator
        Data generator (should have shuffle=False).
    class_names : list of str, optional
        Class names in index order.

    Returns
    -------
    dict with keys:
        accuracy, top3_accuracy, confusion_matrix,
        classification_report, per_class_accuracy
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Reset generator to start
    generator.reset()

    # Get predictions
    logger.info("Running inference on %d samples...", generator.samples)
    y_pred_proba = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = generator.classes

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Top-3 accuracy
    top3_acc = top_k_accuracy_score(y_true, y_pred_proba, k=3)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    # Per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[cls] = accuracy_score(y_true[mask], y_pred[mask])

    metrics = {
        "accuracy":              acc,
        "top3_accuracy":         top3_acc,
        "confusion_matrix":      cm,
        "classification_report": report,
        "per_class_accuracy":    per_class_acc,
        "y_true":                y_true,
        "y_pred":                y_pred,
        "y_pred_proba":          y_pred_proba,
    }

    logger.info("Accuracy:      %.4f", acc)
    logger.info("Top-3 Accuracy: %.4f", top3_acc)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=class_names))

    return metrics


def format_metrics_report(metrics: Dict, class_names: Optional[List[str]] = None) -> str:
    """
    Format metrics into a human-readable report string.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics().
    class_names : list of str, optional

    Returns
    -------
    str
    """
    if class_names is None:
        class_names = CLASS_NAMES

    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Overall Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Top-3 Accuracy:        {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)",
        "",
        "Per-Class Accuracy:",
    ]
    for cls, acc in metrics["per_class_accuracy"].items():
        lines.append(f"  {cls:<15} {acc:.4f} ({acc*100:.2f}%)")

    lines += [
        "",
        "Classification Report:",
        "-" * 60,
    ]
    report = metrics["classification_report"]
    for cls in class_names:
        if cls in report:
            r = report[cls]
            lines.append(
                f"  {cls:<15} precision={r['precision']:.3f}  "
                f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  "
                f"support={int(r['support'])}"
            )

    lines += [
        "",
        f"  Macro avg F1:    {report['macro avg']['f1-score']:.4f}",
        f"  Weighted avg F1: {report['weighted avg']['f1-score']:.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)
