"""Stable classification metric serialization."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


class MetricSummary(TypedDict):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    top_3_accuracy: float


def classification_metrics(
    y_true: NDArray[np.integer], probabilities: NDArray[np.floating]
) -> MetricSummary:
    labels = np.arange(probabilities.shape[1])
    predictions = np.argmax(probabilities, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=cast(Any, 0),
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top_3_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=3, labels=labels)),
    }


def detailed_classification_report(
    y_true: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    class_names: tuple[str, ...],
) -> dict[str, object]:
    """Return aggregate, per-class, and confusion evidence."""
    labels = np.arange(len(class_names))
    predictions = np.argmax(probabilities, axis=1)
    _, recall, _, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=labels,
        average=None,
        zero_division=cast(Any, 0),
    )
    recall_values = np.asarray(recall, dtype=np.float64)
    support_values = np.asarray(support, dtype=np.int64)
    return {
        "metrics": classification_metrics(y_true, probabilities),
        "per_class": {
            name: {
                "recall": float(recall_values[index]),
                "support": int(support_values[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }
