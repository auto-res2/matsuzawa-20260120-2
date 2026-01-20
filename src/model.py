"""src/model.py
Model components used in all experiments.  Contains:
• ScoreNet – class-conditional feature diffusion model.
• MAPHead – light attention-pooling classifier.
• adversarial_feature_augment – FGSM in feature space for AC-FroFA baseline.
Defensive checks & assertions are in place so that failures are caught early.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ScoreNet", "MAPHead", "adversarial_feature_augment"]


class ScoreNet(nn.Module):
    """Two-layer MLP score network with timestep & class conditioning."""

    def __init__(self, dim: int, n_classes: int, embed_dim: int = 128):
        if dim <= 0 or n_classes <= 0:
            raise ValueError("`dim` and `n_classes` must be positive integers.")
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.class_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + embed_dim, 1024),
            nn.SiLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, dim),
        )

    def forward(self, feats: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if feats.dim() != 2:
            raise ValueError("ScoreNet expects input (B, C)")
        if t.dim() == 1:
            t = t.unsqueeze(1)
        h = torch.cat([feats, self.time_embed(t) + self.class_embed(y)], dim=-1)
        return self.net(h)


class MAPHead(nn.Module):
    """Multi-head attention pooling followed by a linear classifier."""

    def __init__(self, dim_in: int, dim_h: int, n_cls: int):
        if dim_in <= 0 or n_cls <= 0:
            raise ValueError("`dim_in` and `n_cls` must be positive integers.")
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_in, num_heads=4, batch_first=True)
        self.fc = nn.Linear(dim_in, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 2:  # (B, C) – add dummy sequence dim for attention
            x = x.unsqueeze(1)
        v, _ = self.attn(x, x, x)
        return self.fc(v.mean(1))


# -----------------------------------------------------------------------------
# Baseline adversarial feature augmentation (AC-FroFA)
# -----------------------------------------------------------------------------

def adversarial_feature_augment(
    feats: torch.Tensor, labels: torch.Tensor, classifier: nn.Module, *, eps: float = 0.5
) -> torch.Tensor:
    """FGSM-style perturbation in frozen feature space."""
    feats_adv = feats.detach().clone().requires_grad_(True)
    logits = classifier(feats_adv)
    loss = F.cross_entropy(logits, labels)
    (grad,) = torch.autograd.grad(loss, feats_adv, create_graph=False)
    perturb = eps * grad.sign()
    return (feats_adv + perturb).detach()
