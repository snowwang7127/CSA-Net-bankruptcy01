# CSA-Net: A Cost-Sensitive Attention Framework for Interpretable Imbalanced Tabular Classification

Reference implementation for the paper *"A Cost-Sensitive Attention Framework for
Interpretable Imbalanced Tabular Classification in Financial Distress Prediction."*

CSA-Net couples three components in a single end-to-end architecture:

1. **Feature embedding** — each financial ratio `x_i` is projected by a shared
   linear map into a `d=32` dimensional space (`e_i = W_e x_i + b_e`).
2. **Feature-level attention** — a scalar importance weight per ratio,
   `alpha_i = softmax_i(v^T tanh(W_a e_i + b_a))`, giving `h = sum_i alpha_i e_i`.
   The weights are read directly from the forward pass as per-sample attributions.
3. **Cost-sensitive optimisation** — **Focal Loss** (`gamma=2.0, alpha=0.75`)
   concentrates gradients on hard minority-class samples.

A `256 -> 128 -> 64` trunk (BatchNorm + GELU + Dropout 0.3) and a sigmoid output
complete the network.

## Repository layout

```
src/
  models.py       CSA-Net + ablation variants (MLP, MLP-SMOTE, MLP-FL, MLP-Attn)
  losses.py       Focal Loss
  data.py         Polish/Taiwanese loaders, KNN imputation, z-score, SMOTE, synthetic data
  metrics.py      AUC-ROC, AUC-PR, F1, Recall, G-Mean + threshold selection
  train.py        repeated stratified 5-fold CV (25 runs), early stopping
  baselines.py    LR, RF, XGBoost, LightGBM, TabNet, FT-Transformer wrappers
  stats.py        Wilcoxon signed-rank + Holm correction
  interpret.py    attention importance, DeepSHAP, Spearman concordance
scripts/
  run_experiment.py     full benchmark on one dataset + significance table
  run_hparam_sweep.py   gamma/alpha sensitivity heatmap (Section 4.3)
config.py         default hyperparameters and sweep grids
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (no data download needed)

This runs the entire pipeline on synthetic severely-imbalanced data and prints a
metrics table plus the Wilcoxon-vs-CSA-Net significance test:

```bash
python -m scripts.run_experiment --dataset smoke --quick
```

## Reproducing the paper

### Data

- **Polish Companies Bankruptcy Dataset** — UCI id 365
  (https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data).
  Place the five `*.arff` files in `data/polish/` as `1year.arff` ... `5year.arff`.
- **Taiwanese Bankruptcy Prediction** — UCI id 572
  (https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction).
  Save the CSV as `data/taiwanese.csv`.

### Run

```bash
# Polish, per subset, all models + baselines
python -m scripts.run_experiment --dataset polish --data-dir data/polish --subset 3year --baselines

# Taiwanese (out-of-sample; hyperparameters transferred, no re-tuning)
python -m scripts.run_experiment --dataset taiwanese --csv data/taiwanese.csv --baselines

# Hyperparameter sensitivity (gamma x alpha)
python -m scripts.run_hparam_sweep --dataset polish --data-dir data/polish --subset 3year
```

## Notes on faithful reproduction

- Imputation and standardisation are fit **only** on the training fold of each split;
  SMOTE (for the MLP-SMOTE variant) is applied **only** to the training fold.
- Threshold-dependent metrics use an operating point chosen on a held-out validation
  slice (max minority-F1) and applied unchanged to the test fold.
- Per-fold metric values, selected thresholds, and seeds should be archived alongside
  results; reported numbers are mean +/- std over the 25 runs.
- Exact figures depend on library versions, hardware, and random seeds.

## License

MIT — see [LICENSE](LICENSE).
