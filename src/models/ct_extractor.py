"""MedicalNet-compatible 3D ResNet-18 feature extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CTFeatureExtractor(nn.Module):
    """Headless MONAI 3D ResNet-18 adapted from one to four CT channels."""

    feature_dim = 512

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 4,
        feature_dim: int = 512,
        freeze: bool = True,
        pretrained_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Only resnet18 is supported by this pipeline.")
        if feature_dim != self.feature_dim:
            raise ValueError("ResNet-18 feature_dim must be 512.")

        from monai.networks.nets import resnet18

        # MONAI MedicalNet weights are defined for one-channel 3D inputs and
        # require the classification head to be disabled while loading.
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
        self._adapt_input_channels(in_channels)
        self.frozen = False
        if freeze:
            self.freeze_all()

    def _load_custom_weights(self, path: str) -> None:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )
        cleaned = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in state.items()
        }
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        logger.info(
            "Loaded custom CT weights (%d missing, %d unexpected)",
            len(missing),
            len(unexpected),
        )

    def _adapt_input_channels(self, in_channels: int) -> None:
        old_conv = self.backbone.conv1
        if old_conv.in_channels == in_channels:
            return
        if old_conv.in_channels != 1:
            raise ValueError(
                f"Expected one-channel pretrained conv, got {old_conv.in_channels}."
            )
        new_conv = nn.Conv3d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.repeat(1, in_channels, 1, 1, 1) / in_channels
            )
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.conv1 = new_conv

    def freeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.frozen = True
        self.backbone.eval()

    def unfreeze_layer4(self) -> None:
        """Unfreeze only the final residual stage for low-rate fine-tuning."""
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True
        self.frozen = False
        # Earlier normalization statistics remain fixed.
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = name.startswith("layer4")

    def train(self, mode: bool = True) -> "CTFeatureExtractor":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        else:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)
