"""
Clinical Feature Encoder Module
=================================
Multi-layer perceptron (MLP) encoder for clinical tabular features.
Transforms raw clinical feature vectors into a learned representation
suitable for cross-modal fusion with CT imaging features.
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClinicalEncoder(nn.Module):
    """MLP encoder for clinical tabular features.

    Builds a sequential stack of ``Linear -> LayerNorm -> GELU -> Dropout``
    blocks followed by a final linear projection layer (without activation) that
    outputs a fixed-size representation for downstream fusion.

    Args:
        input_dim: Dimensionality of the raw clinical feature vector.
        hidden_dims: List of hidden layer sizes.  Each entry produces one
            ``Linear -> BN -> Act -> Dropout`` block.
        output_dim: Dimensionality of the output representation.
        dropout: Dropout probability applied after each activation.
        activation: Activation function name.  Supported: ``"relu"``,
            ``"gelu"``, ``"leaky_relu"``, ``"silu"``.
        use_batchnorm: Deprecated compatibility argument. LayerNorm is always
            used because fold training commonly uses small batches.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: List[int] = None,
        output_dim: int = 64,
        dropout: float = 0.3,
        activation: str = "gelu",
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS.keys())}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        act_cls = self._ACTIVATIONS[activation]

        # ---- Build hidden layers ----
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # ---- Final projection (no activation) ----
        self.output_projection = nn.Linear(prev_dim, output_dim)

        # ---- Weight initialization ----
        self._init_weights()

        logger.info(
            f"ClinicalEncoder initialized: {input_dim} -> "
            f"{hidden_dims} -> {output_dim}, "
            f"activation={activation}, dropout={dropout}, "
            "normalization=layernorm"
        )

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode clinical features.

        Args:
            x: Clinical feature tensor of shape ``(B, input_dim)``.

        Returns:
            Encoded representation of shape ``(B, output_dim)``.
        """
        h = self.hidden_layers(x)
        out = self.output_projection(h)
        return out
