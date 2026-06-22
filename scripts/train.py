#!/usr/bin/env python3
"""Train one fold or all folds of the MRD prediction pipeline."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import build_data_bundle
from src.models.mrd_predictor import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import apply_overrides, load_config, save_config, validate_config
from src.utils.logger import Logger

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    name = train_cfg.get("optimizer", "adamw").lower()
    parameters = list(model.non_backbone_parameters())
    kwargs = {
        "lr": float(train_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
    }
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    train_cfg = cfg["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    params = train_cfg.get("scheduler_params", {})
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(params.get("T_max", train_cfg.get("num_epochs", 100))),
            eta_min=float(params.get("eta_min", 1e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get("step_size", 30)),
            gamma=float(params.get("gamma", 0.1)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg["checkpoint"].get("monitor_mode", "max"),
            factor=float(params.get("factor", 0.1)),
            patience=int(params.get("patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["gated_fusion", "clinical_only", "ct_only"],
        default=None,
    )
    parser.add_argument(
        "--clinical-profile",
        default=None,
        help="Named data.clinical_feature_profiles entry.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--override", nargs="*", default=None)
    return parser.parse_args()


def train_fold(base_cfg: dict, fold: int, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if args.variant:
        cfg["model"]["variant"] = args.variant
    if args.clinical_profile:
        cfg["data"]["active_clinical_profile"] = args.clinical_profile
    validate_config(cfg)
    seed = int(cfg["data"].get("random_seed", 42)) + fold
    set_seed(seed)

    variant = cfg["model"]["variant"]
    profile = cfg["data"].get("active_clinical_profile", "default")
    experiment_name = (
        variant if variant == "ct_only" else f"{variant}_{profile}"
    )
    output_dir = Path(args.output_dir) / experiment_name / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["log_dir"] = str(output_dir / "logs")
    save_config(cfg, str(output_dir / "effective_config.yaml"))

    bundle = build_data_bundle(cfg, fold)
    pd.DataFrame(bundle.split_metadata["manifest"]).to_csv(
        output_dir / "fold_manifest.csv", index=False
    )
    model = build_model(
        cfg,
        bundle.clinical_input_dim,
        load_pretrained=args.resume is None,
    )
    criterion = build_loss(cfg, bundle.train_positive_weight)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    exp_logger = Logger(cfg)
    exp_logger.log_config(cfg)
    try:
        trainer = Trainer(
            model=model,
            train_loader=bundle.loaders["train"],
            val_loader=bundle.loaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            exp_logger=exp_logger,
            clinical_processor=bundle.clinical_processor,
            split_metadata=bundle.split_metadata,
            output_dir=str(output_dir),
        )
        return trainer.fit(args.resume)
    finally:
        exp_logger.close()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    validate_config(cfg)
    n_splits = int(cfg["data"]["cross_validation"].get("n_splits", 5))
    folds = range(n_splits) if args.all_folds else [args.fold]
    if args.resume and args.all_folds:
        raise ValueError("--resume can only be used with one --fold.")
    results = {fold: train_fold(cfg, fold, args) for fold in folds}
    logger.info("Training results: %s", results)


if __name__ == "__main__":
    main()
