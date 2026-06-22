"""Compact gated fusion for small-cohort multimodal learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Project both modalities and learn a feature-wise convex combination."""

    def __init__(
        self,
        ct_feature_dim: int = 512,
        clinical_feature_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.ct_projection = nn.Sequential(
            nn.Linear(ct_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clinical_projection = nn.Sequential(
            nn.Linear(clinical_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(projection_dim * 2, projection_dim)
        self.output_norm = nn.LayerNorm(projection_dim)

    def forward(
        self, ct_features: torch.Tensor, clinical_features: torch.Tensor
    ) -> torch.Tensor:
        ct = self.ct_projection(ct_features)
        clinical = self.clinical_projection(clinical_features)
        gate = torch.sigmoid(self.gate(torch.cat([ct, clinical], dim=-1)))
        return self.output_norm(gate * ct + (1.0 - gate) * clinical)
