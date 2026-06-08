"""
CSA-Net (Cost-Sensitive Attention Network) and ablation variants.

Architecture (matches the paper):
    1. Feature embedding   : each scalar ratio x_i -> e_i = W_e x_i + b_e,
                             with W_e in R^{d x 1}, b_e in R^d, *shared* across all features.
    2. Feature attention   : alpha_i = softmax_i( v^T tanh(W_a e_i + b_a) ),
                             W_a in R^{(d/2) x d}, v in R^{d/2}; scalar weight per ratio.
    3. Attended repr.      : h = sum_i alpha_i e_i  in R^d.
    4. Deep extraction     : h -> 256 -> 128 -> 64, each {Linear, BatchNorm, GELU, Dropout(p)}.
    5. Output              : p_hat = sigmoid(w^T h^(3) + b0).

Ablation variants share the 256-128-64 trunk and differ only in:
    - whether the embedding+attention front-end is used (use_attention),
    - the training loss (BCE vs Focal Loss, chosen in the training script),
    - whether SMOTE is applied to the training fold (handled in the data pipeline).

    MLP        : use_attention=False, BCE,   no SMOTE
    MLP-SMOTE  : use_attention=False, BCE,   SMOTE
    MLP-FL     : use_attention=False, Focal, no SMOTE
    MLP-Attn   : use_attention=True,  BCE,   no SMOTE
    CSA-Net    : use_attention=True,  Focal, no SMOTE
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Shared per-feature linear embedding: (B, F) -> (B, F, d)."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        # A single Linear(1, d) applied independently to every scalar feature.
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F)
        x = x.unsqueeze(-1)            # (B, F, 1)
        return self.proj(x)            # (B, F, d) -- params shared across F


class FeatureAttention(nn.Module):
    """Scalar attention weight per feature: (B, F, d) -> (h: (B, d), alpha: (B, F))."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        hidden = max(1, embed_dim // 2)
        self.W_a = nn.Linear(embed_dim, hidden)       # W_a in R^{(d/2) x d}, b_a
        self.v = nn.Linear(hidden, 1, bias=False)     # v   in R^{d/2}

    def forward(self, e: torch.Tensor):               # e: (B, F, d)
        score = self.v(torch.tanh(self.W_a(e)))       # (B, F, 1)
        alpha = torch.softmax(score, dim=1)           # softmax over the F features
        h = (alpha * e).sum(dim=1)                    # (B, d)
        return h, alpha.squeeze(-1)                   # h:(B,d), alpha:(B,F)


class DeepHead(nn.Module):
    """256 -> 128 -> 64 trunk with BN + GELU + Dropout, then a sigmoid logit."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim, 256, 128, 64]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(64, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(h)).squeeze(-1)    # raw logit (B,)


class CSANet(nn.Module):
    """
    Unified model covering CSA-Net and all neural ablation variants.

    Parameters
    ----------
    n_features : int      number of input financial ratios (64 Polish / 95 Taiwanese)
    use_attention : bool  if True, use the embedding+attention front-end (CSA-Net, MLP-Attn);
                          if False, feed raw features straight into the deep head (MLP*, MLP-FL).
    embed_dim : int       embedding dimension d (default 32)
    dropout : float       dropout probability (default 0.3)
    """

    def __init__(self, n_features: int, use_attention: bool = True,
                 embed_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.embedding = FeatureEmbedding(embed_dim)
            self.attention = FeatureAttention(embed_dim)
            head_in = embed_dim
        else:
            self.embedding = None
            self.attention = None
            head_in = n_features
        self.head = DeepHead(head_in, dropout=dropout)
        self._last_alpha = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            e = self.embedding(x)             # (B, F, d)
            h, alpha = self.attention(e)      # (B, d), (B, F)
            self._last_alpha = alpha
        else:
            h = x
            self._last_alpha = None
        return self.head(h)                   # logits (B,)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample attention vector alpha (B, F). Only valid if use_attention."""
        if not self.use_attention:
            raise RuntimeError("This variant has no attention module.")
        self.eval()
        e = self.embedding(x)
        _, alpha = self.attention(e)
        return alpha


# Convenience factory matching the ablation table -------------------------------------------
VARIANTS = {
    "MLP":       dict(use_attention=False),   # + BCE,   no SMOTE
    "MLP-SMOTE": dict(use_attention=False),   # + BCE,   SMOTE (applied in data pipeline)
    "MLP-FL":    dict(use_attention=False),   # + Focal, no SMOTE
    "MLP-Attn":  dict(use_attention=True),    # + BCE,   no SMOTE
    "CSA-Net":   dict(use_attention=True),    # + Focal, no SMOTE
}

USES_FOCAL = {"MLP": False, "MLP-SMOTE": False, "MLP-FL": True,
              "MLP-Attn": False, "CSA-Net": True}
USES_SMOTE = {"MLP": False, "MLP-SMOTE": True, "MLP-FL": False,
              "MLP-Attn": False, "CSA-Net": False}


def build_variant(name: str, n_features: int, embed_dim: int = 32,
                  dropout: float = 0.3) -> CSANet:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Choices: {list(VARIANTS)}")
    return CSANet(n_features=n_features, embed_dim=embed_dim, dropout=dropout,
                  **VARIANTS[name])
