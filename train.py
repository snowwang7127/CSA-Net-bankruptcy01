"""Training loop and repeated stratified cross-validation.

Protocol (matches the paper):
    - stratified 5-fold CV repeated 5 times with different seeds = 25 runs
    - within each split: an inner validation slice of the training fold is used for
      early stopping (on AUC-ROC) and for operating-threshold selection
    - Adam + cosine LR decay, batch size 256, early stopping (patience 20)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .models import build_variant, USES_FOCAL, USES_SMOTE
from .losses import make_loss
from .data import preprocess_fold, apply_smote
from .metrics import select_threshold, compute_metrics


def _to_loader(X, y, batch_size, shuffle, device):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X, y)
    # drop_last avoids a singleton final batch crashing BatchNorm during training
    drop_last = shuffle and len(X) > batch_size
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       drop_last=drop_last)


@torch.no_grad()
def _predict_proba(model, X, device, batch_size=512):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_one(model, X_tr, y_tr, X_val, y_val, *, use_focal, cfg, device):
    model.to(device)
    loss_fn = make_loss(use_focal, gamma=cfg["gamma"], alpha=cfg["alpha"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    loader = _to_loader(X_tr, y_tr, cfg["batch_size"], True, device)

    best_auc, best_state, patience = -1.0, None, 0
    from sklearn.metrics import roc_auc_score
    for _ in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        # early stopping on inner-validation AUC-ROC
        p_val = _predict_proba(model, X_val, device)
        auc = roc_auc_score(y_val, p_val)
        if auc > best_auc + 1e-5:
            best_auc, best_state, patience = auc, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def cross_validate(variant, X, y, cfg, device="cpu"):
    """Return a list of per-fold metric dicts (len = n_repeats * n_folds)."""
    per_fold = []
    use_focal = USES_FOCAL[variant]
    use_smote = USES_SMOTE[variant]
    n_features = X.shape[1]

    for rep in range(cfg["n_repeats"]):
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                              random_state=cfg["seed"] + rep)
        for tr_idx, te_idx in skf.split(X, y):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]

            # leakage-safe preprocessing fit on the (outer) training fold
            X_tr, X_te = preprocess_fold(X_tr_raw, X_te_raw, knn_k=cfg["knn_k"])

            # inner split for early stopping + threshold selection
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr,
                random_state=cfg["seed"] + rep)

            if use_smote:
                X_fit, y_fit = apply_smote(X_fit, y_fit, seed=cfg["seed"] + rep)

            torch.manual_seed(cfg["seed"] + rep)
            model = build_variant(variant, n_features,
                                  embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])
            model = train_one(model, X_fit, y_fit, X_val, y_val,
                              use_focal=use_focal, cfg=cfg, device=device)

            p_val = _predict_proba(model, X_val, device)
            thr = select_threshold(y_val, p_val)
            p_te = _predict_proba(model, X_te, device)
            per_fold.append(compute_metrics(y_te, p_te, thr))
    return per_fold
