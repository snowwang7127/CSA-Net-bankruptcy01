"""Baseline models: LR, RF, XGBoost, LightGBM, TabNet, FT-Transformer.

Each baseline is evaluated under the same repeated stratified CV, leakage-safe
preprocessing, and threshold-selection protocol as the neural variants. Ensemble
baselines use widely used class-imbalance settings (scale_pos_weight / class_weight
/ is_unbalance) rather than per-subset tuning; the deep tabular baselines use
class-weighted loss and the same validation folds. Heavy dependencies are imported
lazily so the core neural pipeline runs without them.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import preprocess_fold
from .metrics import select_threshold, compute_metrics


def _proba(clf, X):
    p = clf.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


def _make_baseline(name: str, pos_weight: float, seed: int):
    name = name.lower()
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced"), "sklearn"
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=seed, n_jobs=-1), "sklearn"
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight, eval_metric="aucpr",
                             random_state=seed, n_jobs=-1), "sklearn"
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                              is_unbalance=True, random_state=seed, n_jobs=-1), "sklearn"
    if name == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(seed=seed, verbose=0), "tabnet"
    if name == "ft-transformer":
        # Optional: requires `rtdl`. Returns a thin wrapper exposing predict_proba.
        from .ft_transformer import FTTransformerClassifier
        return FTTransformerClassifier(seed=seed, pos_weight=pos_weight), "sklearn"
    raise KeyError(f"Unknown baseline '{name}'")


def cross_validate_baseline(name: str, X, y, cfg):
    per_fold = []
    for rep in range(cfg["n_repeats"]):
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                              random_state=cfg["seed"] + rep)
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = preprocess_fold(X[tr_idx], X[te_idx], knn_k=cfg["knn_k"])
            y_tr, y_te = y[tr_idx], y[te_idx]
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr,
                random_state=cfg["seed"] + rep)
            pos_weight = float((y_fit == 0).sum() / max(1, (y_fit == 1).sum()))
            clf, kind = _make_baseline(name, pos_weight, cfg["seed"] + rep)
            if kind == "tabnet":
                clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                        eval_metric=["auc"], max_epochs=200, patience=20,
                        batch_size=256, weights=1)
            else:
                clf.fit(X_fit, y_fit)
            thr = select_threshold(y_val, _proba(clf, X_val))
            per_fold.append(compute_metrics(y_te, _proba(clf, X_te), thr))
    return per_fold
