"""Default experimental configuration.

Hyperparameters follow the values identified on the Polish 3rd-Year validation
fold and reused unchanged on the Taiwanese benchmark (no re-tuning).
"""

DEFAULT_CONFIG = {
    # model
    "embed_dim": 32,
    "dropout": 0.3,
    # focal loss
    "gamma": 2.0,
    "alpha": 0.75,
    # optimisation
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
    # preprocessing
    "knn_k": 5,
    # cross-validation
    "n_folds": 5,
    "n_repeats": 5,     # 5 x 5 = 25 runs
    "seed": 42,
}

# Hyperparameter sweep grid for the sensitivity analysis (Section 4.3).
GAMMA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]
ALPHA_GRID = [0.25, 0.50, 0.75, 0.90]
EMBED_GRID = [8, 16, 32, 64, 128]
