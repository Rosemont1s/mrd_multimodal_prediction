"""MedicalNet-compatible phase-wise 3D ResNet-18 feature extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class CTFeatureExtractor(nn.Module):
    """Encode each CT phase with one shared single-channel MedicalNet backbone.

    MedicalNet is pretrained on single-channel volumes. Applying the shared
    encoder independently to each registered phase preserves that input
    contract and avoids reducing multiphase CT to a frozen channel average.
    """

    feature_dim = 512

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 4,
        feature_dim: int = 512,
        freeze: bool = True,
        pretrained_weights_path: Optional[str] = None,
        phase_fusion: str = "attention",
        phase_attention_dim: int = 128,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Only resnet18 is supported by this pipeline.")
        if feature_dim != self.feature_dim:
            raise ValueError("ResNet-18 feature_dim must be 512.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if phase_fusion not in {"attention", "mean"}:
            raise ValueError("phase_fusion must be 'attention' or 'mean'.")

        from monai.networks.nets import resnet18

        self.num_phases = in_channels
        self.phase_fusion = phase_fusion
        self.gradient_checkpointing = gradient_checkpointing
        self.backbone = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True,
            pretrained=pretrained and pretrained_weights_path is None,
        )
        if pretrained_weights_path:
            self._load_custom_weights(pretrained_weights_path)

        if phase_fusion == "attention":
            self.phase_embeddings = nn.Parameter(
                torch.zeros(in_channels, self.feature_dim)
            )
            self.phase_attention = nn.Sequential(
                nn.Linear(self.feature_dim, phase_attention_dim),
                nn.Tanh(),
                nn.Linear(phase_attention_dim, 1),
            )
            nn.init.normal_(self.phase_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("phase_embeddings", None)
            self.phase_attention = None

        self.frozen = False
        if freeze:
            self.freeze_all()

    def _load_custom_weights(self, path: str) -> None:
        checkpoint_data = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
        state = checkpoint_data.get(
            "model_state_dict",
            checkpoint_data.get("state_dict", checkpoint_data),
        )
        cleaned = {
            key.removeprefix("module.")
            .removeprefix("ct_extractor.")
            .removeprefix("backbone."): value
            for key, value in state.items()
        }
        backbone_state = self.backbone.state_dict()
        compatible = {
            key: value
            for key, value in cleaned.items()
            if key in backbone_state and backbone_state[key].shape == value.shape
        }
        if not compatible:
            raise ValueError(f"No compatible CT backbone weights found in {path}.")
        missing, unexpected = self.backbone.load_state_dict(
            compatible, strict=False
        )
        logger.info(
            "Loaded %d custom CT tensors (%d missing, %d unexpected or incompatible)",
            len(compatible),
            len(missing),
            len(cleaned) - len(compatible) + len(unexpected),
        )

    def freeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.frozen = True
        self.backbone.eval()

    def unfreeze_last_block(self) -> None:
        """Unfreeze only the final ResNet block for conservative adaptation."""
        for parameter in self.backbone.layer4[-1].parameters():
            parameter.requires_grad = True
        self.frozen = False
        self._freeze_batch_norm_statistics()

    def unfreeze_layer4(self) -> None:
        """Unfreeze the complete final residual stage for low-rate tuning."""
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True
        self.frozen = False
        self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self) -> None:
        """Keep pretrained batch-normalization running statistics fixed."""
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> "CTFeatureExtractor":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        else:
            self._freeze_batch_norm_statistics()
        return self

    def _encode_phase(self, phase: torch.Tensor) -> torch.Tensor:
        trainable_backbone = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        if (
            self.gradient_checkpointing
            and self.training
            and trainable_backbone
        ):
            features = checkpoint(
                self.backbone, phase, use_reentrant=False
            )
        elif trainable_backbone:
            features = self.backbone(phase)
        else:
            with torch.no_grad():
                features = self.backbone(phase)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError("CT input must have shape (B, phases, D, H, W).")
        if x.shape[1] != self.num_phases:
            raise ValueError(
                f"Expected {self.num_phases} CT phases, got {x.shape[1]}."
            )

        phase_features = torch.stack(
            [
                self._encode_phase(x[:, index : index + 1])
                for index in range(self.num_phases)
            ],
            dim=1,
        )
        if self.phase_fusion == "attention":
            attended = phase_features + self.phase_embeddings.unsqueeze(0)
            scores = self.phase_attention(attended).squeeze(-1)
            attention = torch.softmax(scores, dim=1)
        else:
            attention = phase_features.new_full(
                (x.shape[0], self.num_phases), 1.0 / self.num_phases
            )
        fused = torch.sum(phase_features * attention.unsqueeze(-1), dim=1)
        return (fused, attention) if return_attention else fused

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)
