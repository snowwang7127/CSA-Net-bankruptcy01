"""Dataset loading and leakage-safe preprocessing.

Supports:
    - Polish Companies Bankruptcy Dataset  (UCI id 365, five .arff subsets 1year..5year)
    - Taiwanese Bankruptcy Prediction       (UCI id 572, single .csv)
    - synthetic imbalanced data             (--smoke; no download needed)

Preprocessing is applied *inside* each CV fold to prevent leakage:
    KNN imputation (k=5)  -> fit on train fold only
    z-score standardise   -> fit on train fold only
    SMOTE (optional)      -> applied to the train fold only
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

POLISH_SUBSETS = ["1year", "2year", "3year", "4year", "5year"]


# ----------------------------------------------------------------------------- loaders
def load_polish(data_dir: str, subset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one Polish subset from its .arff file. Drops the year indicator if present.

    Download: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
    Expected file: {data_dir}/{subset}.arff
    """
    from scipy.io import arff
    path = os.path.join(data_dir, f"{subset}.arff")
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)
    # class column is the last attribute, byte-encoded b'0'/b'1'
    y = df.iloc[:, -1].apply(lambda v: int(v.decode() if isinstance(v, bytes) else v)).values
    X = df.iloc[:, :-1].astype(float).values
    return X, y


def load_taiwanese(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the Taiwanese dataset. Target column is 'Bankrupt?'.

    Download: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
    """
    df = pd.read_csv(csv_path)
    target = "Bankrupt?" if "Bankrupt?" in df.columns else df.columns[0]
    y = df[target].astype(int).values
    X = df.drop(columns=[target]).astype(float).values
    return X, y


def make_synthetic(n: int = 8000, n_features: int = 64, pos_rate: float = 0.05,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic severely-imbalanced tabular data for smoke testing the full pipeline."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos
    # a handful of informative directions; the rest is noise
    w = rng.normal(size=n_features)
    w[10:] *= 0.15
    X = rng.normal(size=(n, n_features))
    logits = X @ w
    # place positives in the upper tail of the score, with overlap (hard problem)
    order = np.argsort(logits)
    y = np.zeros(n, dtype=int)
    y[order[-n_pos:]] = 1
    flip = rng.random(n) < 0.15            # label noise -> realistic, non-separable
    y[flip] = 1 - y[flip]
    # inject missing values into ~3% of entries (exercises the KNN imputer)
    mask = rng.random(X.shape) < 0.03
    X[mask] = np.nan
    return X, y


# ----------------------------------------------------------------- per-fold preprocessing
def preprocess_fold(X_tr, X_te, knn_k: int = 5):
    """Fit imputer+scaler on train, transform both. Returns (X_tr, X_te)."""
    imp = KNNImputer(n_neighbors=knn_k)
    has_nan = np.isnan(X_tr).any()
    if has_nan:
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def apply_smote(X_tr, y_tr, seed: int = 0):
    """Balance the training fold with SMOTE (used only by the MLP-SMOTE variant)."""
    from imblearn.over_sampling import SMOTE
    k = min(5, max(1, int((y_tr == 1).sum()) - 1))
    X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
    return X_res.astype(np.float32), y_res.astype(int)
