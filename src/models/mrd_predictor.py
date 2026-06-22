"""Configurable CT-only, clinical-only, and gated multimodal predictors."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn

from src.models.clinical_encoder import ClinicalEncoder
from src.models.ct_extractor import CTFeatureExtractor
from src.models.fusion import GatedFusion

logger = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] | None = None,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64]
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MRDPredictor(nn.Module):
    """Binary MRD predictor with mandatory ablation-compatible variants."""

    VARIANTS = {"gated_fusion", "clinical_only", "ct_only"}

    def __init__(
        self,
        cfg: Dict[str, Any],
        clinical_input_dim: int,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = cfg.get("model", {}).get("variant", "gated_fusion")
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unsupported model variant: {self.variant}")
        if clinical_input_dim <= 0 and self.variant != "ct_only":
            raise ValueError("clinical_input_dim must be positive.")

        projection_dim = int(cfg["fusion"].get("projection_dim", 128))
        ct_cfg = cfg["ct_extractor"]
        clinical_cfg = cfg["clinical_encoder"]
        classifier_cfg = cfg["classifier"]

        self.ct_extractor = None
        if self.variant != "clinical_only":
            extractor_cfg = dict(ct_cfg)
            if not load_pretrained:
                extractor_cfg["pretrained"] = False
                extractor_cfg["pretrained_weights_path"] = None
            self.ct_extractor = CTFeatureExtractor(**extractor_cfg)

        self.clinical_encoder = None
        if self.variant != "ct_only":
            self.clinical_encoder = ClinicalEncoder(
                input_dim=clinical_input_dim,
                hidden_dims=clinical_cfg.get("hidden_dims", [128, 64]),
                output_dim=clinical_cfg.get("output_dim", 128),
                dropout=clinical_cfg.get("dropout", 0.3),
                activation="gelu",
                use_batchnorm=False,
            )

        if self.variant == "gated_fusion":
            self.fusion = GatedFusion(
                ct_feature_dim=ct_cfg.get("feature_dim", 512),
                clinical_feature_dim=clinical_cfg.get("output_dim", 128),
                projection_dim=projection_dim,
                dropout=cfg["fusion"].get("dropout", 0.2),
            )
        elif self.variant == "ct_only":
            self.fusion = nn.Sequential(
                nn.Linear(ct_cfg.get("feature_dim", 512), projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(clinical_cfg.get("output_dim", 128), projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )

        if int(classifier_cfg.get("input_dim", 128)) != projection_dim:
            raise ValueError("Classifier input dimension must equal fusion projection.")
        self.classifier = ClassifierHead(
            input_dim=projection_dim,
            hidden_dims=classifier_cfg.get("hidden_dims", [64]),
            num_classes=classifier_cfg.get("num_classes", 1),
            dropout=classifier_cfg.get("dropout", 0.3),
        )

    def forward(
        self, ct_images: torch.Tensor, clinical_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.variant == "gated_fusion":
            ct = self.ct_extractor(ct_images)
            clinical = self.clinical_encoder(clinical_features)
            fused = self.fusion(ct, clinical)
        elif self.variant == "ct_only":
            fused = self.fusion(self.ct_extractor(ct_images))
        else:
            fused = self.fusion(self.clinical_encoder(clinical_features))
        logits = self.classifier(fused)
        return {"logits": logits, "probs": torch.sigmoid(logits)}

    def unfreeze_ct_layer4(self) -> bool:
        if self.ct_extractor is None:
            return False
        self.ct_extractor.unfreeze_layer4()
        return True

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        if self.ct_extractor is None:
            return []
        return (
            parameter
            for parameter in self.ct_extractor.parameters()
            if parameter.requires_grad
        )

    def non_backbone_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and not name.startswith("ct_extractor."):
                yield parameter

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(
    cfg: Dict[str, Any],
    clinical_input_dim: int,
    load_pretrained: bool = True,
) -> MRDPredictor:
    """Build a dimension-validated model using the fold's fitted processor."""
    model = MRDPredictor(cfg, clinical_input_dim, load_pretrained=load_pretrained)
    counts = model.count_parameters()
    logger.info(
        "Built %s model: total=%d trainable=%d frozen=%d",
        model.variant,
        counts["total"],
        counts["trainable"],
        counts["frozen"],
    )
    return model
