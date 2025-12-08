# src/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_proba):
    """
    y_true: 1D array-like of true labels (ints)
    y_pred: 1D array-like of predicted labels (ints)
    y_proba: 2D array-like of predicted probabilities, shape [N, num_classes]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    acc = accuracy_score(y_true, y_pred)

    # macro-average precision/recall (treating all classes equally)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Multi-class AUC using one-vs-rest
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except ValueError:
        # AUC can fail if some classes are missing in y_true
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }
