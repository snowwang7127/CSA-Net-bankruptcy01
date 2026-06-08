"""Evaluation metrics for severely imbalanced binary classification.

Reports the five complementary metrics used in the paper:
    AUC-ROC, AUC-PR, minority-class F1, minority-class Recall, G-Mean.

Threshold-dependent metrics (F1, Recall, G-Mean) use an operating threshold
selected on the validation fold by maximising minority-class F1 over a grid in
[0.05, 0.95], then applied unchanged to the test fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             recall_score, roc_auc_score, confusion_matrix)


def select_threshold(y_val: np.ndarray, p_val: np.ndarray,
                     grid=None) -> float:
    """Pick the probability cutoff that maximises minority-class F1 on validation."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sqrt(sensitivity * specificity)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(sens * spec))


def compute_metrics(y_true: np.ndarray, p_prob: np.ndarray,
                    threshold: float) -> dict:
    """All five metrics for one fold given probabilities and a fixed threshold."""
    y_pred = (p_prob >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, p_prob),
        "auc_pr": average_precision_score(y_true, p_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "g_mean": g_mean(y_true, y_pred),
    }


def summarise(per_fold: list[dict]) -> dict:
    """mean +/- std across folds for each metric."""
    keys = per_fold[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_fold], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out
