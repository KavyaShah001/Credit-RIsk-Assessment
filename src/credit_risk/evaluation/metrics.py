"""Binary default-proxy metrics for imbalanced classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return probability and threshold metrics with default as the positive class."""
    truth = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    predictions = (probs >= threshold).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "roc_auc": float(roc_auc_score(truth, probs)),
        "pr_auc_average_precision": float(average_precision_score(truth, probs)),
        "brier_score": float(brier_score_loss(truth, probs)),
        "threshold": float(threshold),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "accuracy_supplementary": float(accuracy_score(truth, predictions)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
