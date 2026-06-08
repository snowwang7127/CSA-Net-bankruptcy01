"""Dual interpretability: attention weights, DeepSHAP, and their rank concordance.

For a trained CSA-Net instance:
    - global attention importance = mean over samples of alpha_i
    - global SHAP importance       = mean |phi_i| over samples (DeepSHAP)
    - concordance                  = Spearman rho over the top-k features
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def attention_importance(model, X, device="cpu") -> np.ndarray:
    """Mean attention weight per feature across the provided samples (F,)."""
    X = torch.as_tensor(X, dtype=torch.float32).to(device)
    alpha = model.attention_weights(X).cpu().numpy()   # (N, F)
    return alpha.mean(axis=0)


def shap_importance(model, X_background, X_explain, device="cpu") -> np.ndarray:
    """Mean |SHAP| per feature via DeepExplainer. Requires the `shap` package."""
    import shap
    model.eval().to(device)
    bg = torch.as_tensor(X_background, dtype=torch.float32).to(device)
    xe = torch.as_tensor(X_explain, dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model, bg)
    vals = explainer.shap_values(xe, check_additivity=False)
    if isinstance(vals, list):
        vals = vals[0]
    return np.abs(vals).mean(axis=0)                   # (F,)


def rank_concordance(att_imp: np.ndarray, shap_imp: np.ndarray, top_k: int = 15):
    """Spearman rho between attention and SHAP rankings over the top-k SHAP features."""
    top = np.argsort(shap_imp)[::-1][:top_k]
    rho, p = spearmanr(att_imp[top], shap_imp[top])
    return float(rho), float(p), top.tolist()
