"""Fold trainer with staged CT fine-tuning and reproducible checkpoints."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import MetricComputer

logger = logging.getLogger(__name__)


def _package_versions() -> Dict[str, str]:
    versions = {}
    for package in ("torch", "monai", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        cfg: Dict[str, Any],
        exp_logger: Any,
        clinical_processor: Any,
        split_metadata: Dict[str, Any],
        output_dir: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.exp_logger = exp_logger
        self.clinical_processor = clinical_processor
        self.split_metadata = split_metadata
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["training"]
        self.num_epochs = int(train_cfg.get("num_epochs", 100))
        self.stage2_start_epoch = int(train_cfg.get("stage2_start_epoch", 10))
        self.stage3_start_epoch = int(train_cfg.get("stage3_start_epoch", 30))
        self.grad_clip_max_norm = float(train_cfg.get("grad_clip_max_norm", 0.0))
        self.patience = int(train_cfg.get("early_stopping_patience", 15))
        self.best_metric: Optional[float] = None
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.stage2_active = False
        self.stage3_active = False

        ckpt_cfg = cfg["checkpoint"]
        self.monitor_metric = ckpt_cfg.get("monitor_metric", "val_auroc")
        self.monitor_mode = ckpt_cfg.get("monitor_mode", "max")
        self.metrics_list = cfg["evaluation"]["metrics"]
        self.threshold = float(cfg["evaluation"].get("threshold", 0.5))
        self.device = self._resolve_device(cfg)
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.last_val_predictions: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
        accelerator = cfg["device"].get("accelerator", "auto")
        if accelerator == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if accelerator == "cuda":
            gpu_id = cfg["device"].get("gpu_ids", [0])[0]
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        meter = MetricComputer(self.metrics_list, self.threshold)
        running_loss = 0.0
        records = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc="train" if training else "validate", leave=False):
                ct = batch["ct"].to(self.device, non_blocking=True)
                clinical = batch["clinical"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True).float()
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                autocast = (
                    torch.amp.autocast("cuda")
                    if self.use_amp
                    else torch.amp.autocast("cpu", enabled=False)
                )
                with autocast:
                    outputs = self.model(ct, clinical)
                    loss = self.criterion(outputs["logits"], labels)

                if training:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip_max_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip_max_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip_max_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip_max_norm
                            )
                        self.optimizer.step()

                running_loss += float(loss.item())
                meter.update(outputs["logits"], labels)
                if not training:
                    probabilities = outputs["probs"].detach().cpu().reshape(-1)
                    true_labels = labels.detach().cpu().reshape(-1)
                    phase_attention = outputs.get("phase_attention")
                    if phase_attention is not None:
                        phase_attention = phase_attention.detach().cpu()
                    phase_names = self.cfg["data"].get("ct_sequences", [])
                    for row_index, (
                        patient_id,
                        true_label,
                        probability,
                    ) in enumerate(
                        zip(
                            batch["patient_id"], true_labels, probabilities
                        )
                    ):
                        record = {
                            "patient_id": str(patient_id),
                            "label": int(true_label),
                            "probability": float(probability),
                            "fold": int(self.split_metadata["fold"]),
                        }
                        if phase_attention is not None:
                            record.update(
                                {
                                    f"attention_{phase_name}": float(weight)
                                    for phase_name, weight in zip(
                                        phase_names,
                                        phase_attention[row_index],
                                    )
                                }
                            )
                        records.append(record)

        metrics = meter.compute()
        metrics["loss"] = running_loss / max(len(loader), 1)
        if not training:
            self.last_val_predictions = records
        return metrics

    def _add_new_backbone_parameters(self, group_name: str) -> int:
        existing = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter
            for parameter in self.model.backbone_parameters()
            if id(parameter) not in existing
        ]
        if not new_parameters:
            return 0
        backbone_lr = float(
            self.cfg["training"].get("backbone_learning_rate", 1e-5)
        )
        self.optimizer.add_param_group(
            {"params": new_parameters, "lr": backbone_lr, "name": group_name}
        )
        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs.append(backbone_lr)
        return sum(parameter.numel() for parameter in new_parameters)

    def _activate_stage2(self) -> None:
        if self.stage2_active or not self.model.unfreeze_ct_last_block():
            return
        added = self._add_new_backbone_parameters("ct_layer4_last_block")
        self.stage2_active = True
        self.epochs_without_improvement = 0
        logger.info(
            "Stage 2 activated: final CT residual block unfrozen (%d parameters).",
            added,
        )

    def _activate_stage3(self) -> None:
        if self.stage3_active or not self.model.unfreeze_ct_layer4():
            return
        added = self._add_new_backbone_parameters("ct_layer4_remaining")
        self.stage2_active = True
        self.stage3_active = True
        self.epochs_without_improvement = 0
        logger.info(
            "Stage 3 activated: complete CT layer4 unfrozen (%d new parameters).",
            added,
        )

    def fit(self, resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        best_path = self.output_dir / "best_model.pt"
        history = {"train": [], "val": []}

        for epoch in range(self.start_epoch, self.num_epochs):
            if epoch >= self.stage2_start_epoch:
                self._activate_stage2()
            if epoch >= self.stage3_start_epoch:
                self._activate_stage3()
            started = time.time()
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            monitor_key = self.monitor_metric.removeprefix("val_")
            monitor_value = float(val_metrics.get(monitor_key, 0.0))
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            log_values = {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "lr/head": self.optimizer.param_groups[0]["lr"],
            }
            self.exp_logger.log_scalars(log_values, epoch)
            logger.info(
                "Epoch %d/%d loss=%.4f val_loss=%.4f val_auroc=%.4f time=%.1fs",
                epoch + 1,
                self.num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics.get("auroc", 0.0),
                time.time() - started,
            )

            if self._is_improvement(monitor_value):
                self.best_metric = monitor_value
                self.epochs_without_improvement = 0
                self.save_checkpoint(best_path, epoch, val_metrics)
                pd.DataFrame(self.last_val_predictions).to_csv(
                    self.output_dir / "oof_predictions.csv", index=False
                )
            else:
                self.epochs_without_improvement += 1
            if self.patience and self.epochs_without_improvement >= self.patience:
                logger.info("Early stopping after %d stale epochs.", self.patience)
                break

        return {
            "best_metric": self.best_metric,
            "best_checkpoint": str(best_path),
            "oof_predictions": str(self.output_dir / "oof_predictions.csv"),
            "history": history,
        }

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        return value > self.best_metric if self.monitor_mode == "max" else value < self.best_metric

    def save_checkpoint(
        self, path: str | Path, epoch: int, metrics: Dict[str, float]
    ) -> None:
        processor_path = self.output_dir / "clinical_processor.pkl"
        self.clinical_processor.save(processor_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "scaler_state_dict": (
                    self.scaler.state_dict() if self.scaler is not None else None
                ),
                "metrics": metrics,
                "config": self.cfg,
                "clinical_input_dim": self.clinical_processor.get_feature_dim(),
                "clinical_processor_path": str(processor_path),
                "split_metadata": self.split_metadata,
                "package_versions": _package_versions(),
                "threshold": self.threshold,
                "best_metric": self.best_metric,
                "stage2_active": self.stage2_active,
                "stage3_active": self.stage3_active,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("stage2_active"):
            self._activate_stage2()
        if checkpoint.get("stage3_active"):
            self._activate_stage3()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = checkpoint.get("best_metric")
        return checkpoint
