"""Run the full benchmark on one dataset and print a metrics table + significance.

Examples
--------
    # Smoke test on synthetic data (no download required) -- verifies the repo runs:
    python -m scripts.run_experiment --dataset smoke --quick

    # Polish, one subset:
    python -m scripts.run_experiment --dataset polish --data-dir data/polish --subset 3year

    # Taiwanese:
    python -m scripts.run_experiment --dataset taiwanese --csv data/taiwanese.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from config import DEFAULT_CONFIG
from src.models import VARIANTS
from src.train import cross_validate
from src.metrics import summarise
from src.stats import pairwise_wilcoxon
from src.data import load_polish, load_taiwanese, make_synthetic

NEURAL = list(VARIANTS)                       # MLP, MLP-SMOTE, MLP-FL, MLP-Attn, CSA-Net
BASELINES = ["lr", "rf", "xgboost", "lightgbm", "tabnet", "ft-transformer"]


def get_data(args):
    if args.dataset == "smoke":
        return make_synthetic(n=2000 if args.quick else 8000,
                              n_features=64, pos_rate=0.05, seed=0)
    if args.dataset == "polish":
        return load_polish(args.data_dir, args.subset)
    if args.dataset == "taiwanese":
        return load_taiwanese(args.csv)
    raise ValueError(args.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["smoke", "polish", "taiwanese"], required=True)
    ap.add_argument("--data-dir", default="data/polish")
    ap.add_argument("--subset", default="3year")
    ap.add_argument("--csv", default="data/taiwanese.csv")
    ap.add_argument("--baselines", action="store_true",
                    help="also run LR/RF/XGB/LGBM/TabNet/FT-T if installed")
    ap.add_argument("--quick", action="store_true",
                    help="1 repeat x 3 folds, few epochs (fast sanity check)")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.quick:
        cfg.update(n_repeats=1, n_folds=3, max_epochs=15, patience=5)

    X, y = get_data(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dataset={args.dataset}  X={X.shape}  pos_rate={y.mean():.4f}  device={device}\n")

    scores_auc_pr = {}     # for significance test on the primary imbalanced metric
    rows = []

    for variant in NEURAL:
        per_fold = cross_validate(variant, X, y, cfg, device=device)
        s = summarise(per_fold)
        scores_auc_pr[variant] = np.array([d["auc_pr"] for d in per_fold])
        rows.append((variant, s))

    if args.baselines:
        from src.baselines import cross_validate_baseline
        for name in BASELINES:
            try:
                per_fold = cross_validate_baseline(name, X, y, cfg)
            except Exception as e:                      # missing optional dependency
                print(f"[skip] baseline {name}: {e}")
                continue
            s = summarise(per_fold)
            scores_auc_pr[name] = np.array([d["auc_pr"] for d in per_fold])
            rows.append((name, s))

    # ---- print metrics table ----
    hdr = f"{'model':<16}{'AUC-ROC':>16}{'AUC-PR':>16}{'F1':>16}{'Recall':>16}{'G-Mean':>16}"
    print(hdr); print("-" * len(hdr))
    for name, s in rows:
        def cell(k): m, sd = s[k]; return f"{m:.3f}\u00b1{sd:.3f}"
        print(f"{name:<16}{cell('auc_roc'):>16}{cell('auc_pr'):>16}"
              f"{cell('f1'):>16}{cell('recall'):>16}{cell('g_mean'):>16}")

    # ---- significance vs CSA-Net on AUC-PR ----
    if "CSA-Net" in scores_auc_pr and len(scores_auc_pr) > 1:
        print("\nWilcoxon signed-rank vs CSA-Net on AUC-PR (Holm-corrected):")
        for r in pairwise_wilcoxon(scores_auc_pr, "CSA-Net"):
            flag = "significant" if r["significant"] else "n.s."
            print(f"  CSA-Net vs {r['model']:<14} p_holm={r['p_holm']:.4f}  {flag}")


if __name__ == "__main__":
    main()
