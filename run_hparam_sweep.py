"""Hyperparameter sensitivity sweep over Focal-Loss gamma and class weight alpha
(reproduces the Section 4.3 heatmap). Reports validation-fold AUC-ROC per cell.

    python -m scripts.run_hparam_sweep --dataset smoke --quick
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from config import DEFAULT_CONFIG, GAMMA_GRID, ALPHA_GRID
from src.train import cross_validate
from src.metrics import summarise
from src.data import make_synthetic, load_polish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["smoke", "polish"], default="smoke")
    ap.add_argument("--data-dir", default="data/polish")
    ap.add_argument("--subset", default="3year")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.quick:
        cfg.update(n_repeats=1, n_folds=3, max_epochs=15, patience=5)

    if args.dataset == "smoke":
        X, y = make_synthetic(n=2000 if args.quick else 8000, pos_rate=0.05)
    else:
        X, y = load_polish(args.data_dir, args.subset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'gamma\\\\alpha':>12}" + "".join(f"{a:>10}" for a in ALPHA_GRID))
    for g in GAMMA_GRID:
        cells = []
        for a in ALPHA_GRID:
            c = dict(cfg, gamma=g, alpha=a)
            per_fold = cross_validate("CSA-Net", X, y, c, device=device)
            cells.append(summarise(per_fold)["auc_roc"][0])
        print(f"{g:>12}" + "".join(f"{v:>10.3f}" for v in cells))


if __name__ == "__main__":
    main()
