"""Focal Loss for binary classification (Lin et al., 2017).

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t is the model's probability for the true class and alpha_t is a
class-specific weight. Defaults gamma=2.0, alpha=0.75 follow the paper's
grid-search optimum on the Polish 3rd-Year validation fold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (B,)
        targets = targets.float()
        p = torch.sigmoid(logits)
        # per-sample BCE without reduction
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)            # prob of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(use_focal: bool, gamma: float = 2.0, alpha: float = 0.75) -> nn.Module:
    if use_focal:
        return FocalLoss(gamma=gamma, alpha=alpha)
    return nn.BCEWithLogitsLoss()
