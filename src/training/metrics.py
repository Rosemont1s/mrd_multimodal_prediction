"""
Evaluation Metrics Module
==========================
Provides a ``MetricComputer`` class that accumulates predictions across
batches and computes clinically relevant metrics for binary MRD status
prediction: AUROC, AUPRC, accuracy, sensitivity, specificity, and F1.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricComputer:
    """Accumulates predictions and computes classification metrics.

    Designed for binary classification evaluation.  Call ``update()`` after
    each batch, then ``compute()`` at the end of the epoch to obtain all
    metrics.

    Args:
        metrics_list: List of metric names to compute.  Supported:
            ``"auroc"``, ``"auprc"``, ``"accuracy"``, ``"sensitivity"``,
            ``"specificity"``, ``"f1"``.
        threshold: Decision threshold for converting probabilities to
            binary predictions (used for accuracy, sensitivity, specificity,
            F1).
    """

    SUPPORTED_METRICS = {
        "auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"
    }

    def __init__(
        self,
        metrics_list: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        if metrics_list is None:
            metrics_list = [
                "auroc", "auprc", "accuracy",
                "sensitivity", "specificity", "f1",
            ]

        unsupported = set(metrics_list) - self.SUPPORTED_METRICS
        if unsupported:
            raise ValueError(
                f"Unsupported metrics: {unsupported}. "
                f"Choose from {self.SUPPORTED_METRICS}."
            )

        self.metrics_list = metrics_list
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions and targets."""
        self._all_logits: List[np.ndarray] = []
        self._all_targets: List[np.ndarray] = []

    def update(self, logits: "torch.Tensor", targets: "torch.Tensor") -> None:
        """Accumulate a batch of predictions and targets.

        Args:
            logits: Raw model logits (before sigmoid) — shape ``(B,)`` or
                ``(B, 1)``.
            targets: Ground-truth binary labels — shape ``(B,)`` or ``(B, 1)``.
        """
        import torch

        # Detach from graph and move to CPU
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        self._all_logits.append(logits_np)
        self._all_targets.append(targets_np)

    def compute(self) -> Dict[str, float]:
        """Compute all requested metrics from accumulated predictions.

        Returns:
            Dictionary mapping metric name to its computed value.
            Metrics that cannot be computed (e.g., AUROC with single class)
            will have value ``0.0``.
        """
        if not self._all_logits:
            logger.warning("No predictions accumulated. Returning zeros.")
            return {m: 0.0 for m in self.metrics_list}

        logits = np.concatenate(self._all_logits)
        targets = np.concatenate(self._all_targets)

        # Sigmoid to get probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Binary predictions
        preds = (probs >= self.threshold).astype(np.int32)
        targets_int = targets.astype(np.int32)

        results: Dict[str, float] = {}

        for metric in self.metrics_list:
            if metric == "auroc":
                results["auroc"] = self._safe_auroc(targets, probs)
            elif metric == "auprc":
                results["auprc"] = self._safe_auprc(targets, probs)
            elif metric == "accuracy":
                results["accuracy"] = float(accuracy_score(targets_int, preds))
            elif metric == "sensitivity":
                results["sensitivity"] = self._sensitivity(targets_int, preds)
            elif metric == "specificity":
                results["specificity"] = self._specificity(targets_int, preds)
            elif metric == "f1":
                results["f1"] = float(
                    f1_score(targets_int, preds, zero_division=0)
                )

        return results

    # ------------------------------------------------------------------
    # Internal metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_auroc(targets: np.ndarray, probs: np.ndarray) -> float:
        """Compute AUROC with graceful handling of single-class edge cases."""
        try:
            unique_classes = np.unique(targets)
            if len(unique_classes) < 2:
                logger.warning(
                    "Only one class present in targets — AUROC is undefined. "
                    "Returning 0.0."
                )
                return 0.0
            return float(roc_auc_score(targets, probs))
        except Exception as e:
            logger.warning(f"AUROC computation failed: {e}. Returning 0.0.")
            return 0.0

    @staticmethod
    def _safe_auprc(targets: np.ndarray, probs: np.ndarray) -> float:
        """Compute AUPRC with graceful handling of edge cases."""
        try:
            unique_classes = np.unique(targets)
            if len(unique_classes) < 2:
                logger.warning(
                    "Only one class present in targets — AUPRC is undefined. "
                    "Returning 0.0."
                )
                return 0.0
            return float(average_precision_score(targets, probs))
        except Exception as e:
            logger.warning(f"AUPRC computation failed: {e}. Returning 0.0.")
            return 0.0

    @staticmethod
    def _sensitivity(targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute sensitivity (recall / true positive rate).

        Sensitivity = TP / (TP + FN)
        """
        tp = np.sum((preds == 1) & (targets == 1))
        fn = np.sum((preds == 0) & (targets == 1))
        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    @staticmethod
    def _specificity(targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute specificity (true negative rate).

        Specificity = TN / (TN + FP)
        """
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        if tn + fp == 0:
            return 0.0
        return float(tn / (tn + fp))

    def __str__(self) -> str:
        """Return a formatted string of all computed metrics."""
        results = self.compute()
        lines = []
        for name, value in results.items():
            lines.append(f"  {name:<15s}: {value:.4f}")
        return "Metrics:\n" + "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MetricComputer(metrics={self.metrics_list}, "
            f"threshold={self.threshold})"
        )


def select_operating_threshold(
    targets: np.ndarray, probabilities: np.ndarray, strategy: str = "youden"
) -> float:
    """Select one threshold from pooled out-of-fold predictions."""
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(targets)) < 2:
        raise ValueError("Threshold selection requires both outcome classes.")
    if strategy != "youden":
        raise ValueError("Only the 'youden' threshold strategy is supported.")
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    finite = np.isfinite(thresholds)
    index = int(np.argmax((tpr - fpr)[finite]))
    return float(np.clip(thresholds[finite][index], 0.0, 1.0))


def metrics_from_probabilities(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute scalar metrics and flattened confusion-matrix counts."""
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": MetricComputer._safe_auroc(targets, probabilities),
        "auprc": MetricComputer._safe_auprc(targets, probabilities),
        "accuracy": float(accuracy_score(targets, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_confidence_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Patient-level stratified bootstrap intervals for scalar metrics."""
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(targets == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in by_class):
        raise ValueError("Bootstrap confidence intervals require both classes.")

    samples: Dict[str, List[float]] = {
        key: [] for key in ("auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1")
    }
    for _ in range(n_bootstrap):
        selected = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in by_class]
        )
        result = metrics_from_probabilities(
            targets[selected], probabilities[selected], threshold
        )
        for key in samples:
            samples[key].append(float(result[key]))

    alpha = (1.0 - confidence_level) / 2.0
    return {
        key: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for key, values in samples.items()
    }
