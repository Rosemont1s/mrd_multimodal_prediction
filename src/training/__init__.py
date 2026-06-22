"""
MRD Multimodal Prediction — Training Package
"""
from src.training.metrics import MetricComputer

__all__ = [
    "FocalLoss",
    "WeightedBCELoss",
    "build_loss",
    "MetricComputer",
    "Trainer",
]


def __getattr__(name: str):
    """Load torch-dependent training objects only when requested."""
    if name in {"FocalLoss", "WeightedBCELoss", "build_loss"}:
        from src.training import losses

        return getattr(losses, name)
    if name == "Trainer":
        from src.training.trainer import Trainer

        return Trainer
    raise AttributeError(name)
