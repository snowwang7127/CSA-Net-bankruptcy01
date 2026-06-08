"""Pairwise significance testing with multiple-comparison control.

Two-sided Wilcoxon signed-rank test on the 25 per-fold metric values for each
model pair, with Holm correction across the family of comparisons (alpha=0.05).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm_correction(pvals: list[float], alpha: float = 0.05):
    """Return (reject flags, adjusted p-values) under Holm step-down."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    reject = adj < alpha
    return reject.tolist(), adj.tolist()


def pairwise_wilcoxon(scores: dict[str, np.ndarray], reference: str,
                      alpha: float = 0.05):
    """Compare every model against `reference` on per-fold scores.

    scores : {model_name: array of per-fold values (length 25)}
    Returns a list of dicts with raw/adjusted p-values and significance flags.
    """
    others = [m for m in scores if m != reference]
    raw = []
    for m in others:
        diff = scores[reference] - scores[m]
        if np.allclose(diff, 0):
            raw.append(1.0)
        else:
            raw.append(wilcoxon(scores[reference], scores[m]).pvalue)
    reject, adj = holm_correction(raw, alpha=alpha)
    return [
        {"model": m, "p_raw": raw[i], "p_holm": adj[i], "significant": reject[i]}
        for i, m in enumerate(others)
    ]
