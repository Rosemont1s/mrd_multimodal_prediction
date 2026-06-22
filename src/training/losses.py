"""
Loss Functions Module
======================
Provides loss functions tailored for binary MRD status prediction,
including Focal Loss for class-imbalanced scenarios and a weighted BCE
wrapper.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary Focal Loss for class-imbalanced classification.

    Focal loss down-weights well-classified examples and focuses training
    on hard misclassified samples, which is especially useful when the
    MRD-positive class is underrepresented.

    .. math::

        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    Args:
        alpha: Weighting factor for the positive class. ``alpha`` is applied
            to positive samples, ``(1 - alpha)`` to negatives.
        gamma: Focusing parameter. Higher values penalize easy examples more.
        reduction: Reduction method: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid) of shape ``(B, 1)``.
            targets: Ground-truth labels (0.0 or 1.0) of shape ``(B, 1)``.

        Returns:
            Scalar loss (if reduction is ``"mean"`` or ``"sum"``) or
            per-sample loss of shape ``(B, 1)`` (if ``"none"``).
        """
        # Numerically stable sigmoid
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Compute focal weights
        # p_t = p for positive, (1-p) for negative
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        # Binary cross-entropy (per-element)
        bce = -targets * torch.log(probs) - (1.0 - targets) * torch.log(1.0 - probs)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss.

    Thin wrapper around ``torch.nn.BCEWithLogitsLoss`` that applies a
    ``pos_weight`` to up-weight the positive (MRD+) class.

    Args:
        pos_weight: Multiplicative weight for positive samples.
    """

    def __init__(self, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Raw model outputs of shape ``(B, 1)``.
            targets: Ground-truth labels of shape ``(B, 1)``.

        Returns:
            Scalar loss.
        """
        # Move pos_weight to same device as logits
        if self.criterion.pos_weight.device != logits.device:
            self.criterion.pos_weight = self.criterion.pos_weight.to(logits.device)
        return self.criterion(logits, targets)


def build_loss(
    cfg: Dict[str, Any], train_positive_weight: float | None = None
) -> nn.Module:
    """Factory function to build the loss criterion from configuration.

    Args:
        cfg: Full configuration dictionary.  Reads ``cfg['training']['loss']``
            to determine which loss to instantiate.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If the loss type is not recognized.
    """
    train_cfg = cfg["training"]
    loss_type = train_cfg.get("loss", "focal").lower()

    if loss_type == "auto_weighted_bce":
        if train_positive_weight is None:
            raise ValueError(
                "auto_weighted_bce requires the fold's train_positive_weight."
            )
        loss = WeightedBCELoss(pos_weight=train_positive_weight)
        logger.info(
            "Built auto-weighted BCE: pos_weight=%.4f", train_positive_weight
        )
    elif loss_type == "focal":
        loss = FocalLoss(
            alpha=train_cfg.get("focal_alpha", 0.75),
            gamma=train_cfg.get("focal_gamma", 2.0),
        )
        logger.info(
            f"Built FocalLoss: alpha={loss.alpha}, gamma={loss.gamma}"
        )
    elif loss_type == "weighted_bce":
        pos_weight = train_cfg.get("pos_weight", 2.0)
        loss = WeightedBCELoss(pos_weight=pos_weight)
        logger.info(f"Built WeightedBCELoss: pos_weight={pos_weight}")
    elif loss_type == "bce":
        loss = nn.BCEWithLogitsLoss()
        logger.info("Built standard BCEWithLogitsLoss")
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Choose from: 'focal', 'weighted_bce', 'bce'."
        )

    return loss
