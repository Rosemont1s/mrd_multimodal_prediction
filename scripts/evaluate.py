#!/usr/bin/env python3
"""Evaluate checkpoints, aggregate OOF predictions, or ensemble the test set."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.clinical_processor import ClinicalProcessor
from src.data.dataset import MRDMultimodalDataset
from src.models.mrd_predictor import build_model
from src.training.metrics import (
    bootstrap_confidence_intervals,
    calibration_curve_data,
    decision_curve_data,
    metrics_from_probabilities,
    select_operating_threshold,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--aggregate-oof", action="store_true")
    parser.add_argument("--ensemble-test", action="store_true")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def _device(cfg: dict) -> torch.device:
    accelerator = cfg["device"].get("accelerator", "auto")
    if accelerator == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if accelerator == "cuda":
        return torch.device(f"cuda:{cfg['device'].get('gpu_ids', [0])[0]}")
    return torch.device("cpu")


def _processor_path(checkpoint_path: Path, checkpoint: dict) -> Path:
    configured = Path(checkpoint.get("clinical_processor_path", ""))
    if configured.exists():
        return configured
    sibling = checkpoint_path.parent / "clinical_processor.pkl"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "Fitted clinical processor is missing from checkpoint artifacts."
    )


@torch.no_grad()
def predict_checkpoint(checkpoint_path: Path, split: str) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    processor = ClinicalProcessor.load(_processor_path(checkpoint_path, checkpoint))
    ids = checkpoint["split_metadata"][f"{split}_ids"]
    dataset = MRDMultimodalDataset(ids, cfg, processor, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
    )
    device = _device(cfg)
    model = build_model(
        cfg, int(checkpoint["clinical_input_dim"]), load_pretrained=False
    )
    if checkpoint.get("stage3_active"):
        model.unfreeze_ct_layer4()
    elif checkpoint.get("stage2_active"):
        model.unfreeze_ct_last_block()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    records: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"{checkpoint_path.parent.name}:{split}"):
        outputs = model(
            batch["ct"].to(device, non_blocking=True),
            batch["clinical"].to(device, non_blocking=True),
        )
        probabilities = outputs["probs"].cpu().reshape(-1).numpy()
        labels = batch["label"].reshape(-1).numpy()
        phase_attention = outputs.get("phase_attention")
        if phase_attention is not None:
            phase_attention = phase_attention.cpu()
        phase_names = cfg["data"].get("ct_sequences", [])
        for row_index, (patient_id, label, probability) in enumerate(
            zip(batch["patient_id"], labels, probabilities)
        ):
            record = {
                "patient_id": str(patient_id),
                "label": int(label),
                "probability": float(probability),
                "fold": int(checkpoint["split_metadata"]["fold"]),
            }
            if phase_attention is not None:
                record.update(
                    {
                        f"attention_{phase_name}": float(weight)
                        for phase_name, weight in zip(
                            phase_names, phase_attention[row_index]
                        )
                    }
                )
            records.append(record)
    return pd.DataFrame(records)


def _report(
    predictions: pd.DataFrame, threshold: float, cfg: dict, output_dir: Path, stem: str
) -> dict:
    targets = predictions["label"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    metrics = metrics_from_probabilities(targets, probabilities, threshold)
    intervals = bootstrap_confidence_intervals(
        targets,
        probabilities,
        threshold,
        n_bootstrap=int(cfg["evaluation"].get("bootstrap_samples", 1000)),
        confidence_level=float(cfg["evaluation"].get("confidence_level", 0.95)),
        seed=int(cfg["data"].get("random_seed", 42)),
    )
    calibration = calibration_curve_data(
        targets,
        probabilities,
        n_bins=int(cfg["evaluation"].get("calibration_bins", 10)),
    )
    decision_curve = decision_curve_data(targets, probabilities)
    subgroup_reports = {}
    clinical_path = Path(cfg["data"]["clinical_csv"])
    subgroup_columns = cfg["evaluation"].get("subgroups", [])
    if clinical_path.exists() and subgroup_columns:
        clinical = pd.read_csv(clinical_path, dtype={"patient_id": str})
        available = [
            column for column in subgroup_columns if column in clinical.columns
        ]
        enriched = predictions.merge(
            clinical[["patient_id", *available]],
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        for column in available:
            subgroup_reports[column] = {}
            for value, group in enriched.groupby(column, dropna=False):
                if group["label"].nunique() < 2:
                    continue
                subgroup_reports[column][str(value)] = metrics_from_probabilities(
                    group["label"].to_numpy(),
                    group["probability"].to_numpy(),
                    threshold,
                )
    report = {
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        "confidence_intervals": intervals,
        "calibration": calibration,
        "decision_curve": decision_curve,
        "subgroups": subgroup_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.assign(
        predicted_label=(predictions["probability"] >= threshold).astype(int)
    ).to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    with open(output_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("%s report: %s", stem, report)
    return report


def aggregate_oof(checkpoint_dir: Path, output_dir: Path) -> float:
    paths = sorted(checkpoint_dir.glob("fold_*/oof_predictions.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold OOF files found under {checkpoint_dir}")
    oof = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in paths])
    if oof["patient_id"].duplicated().any():
        raise ValueError("OOF aggregation found duplicate patient predictions.")
    checkpoint = torch.load(
        checkpoint_dir / "fold_0" / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint["config"]
    expected_folds = int(cfg["data"]["cross_validation"].get("n_splits", 5))
    if len(paths) != expected_folds:
        raise ValueError(f"Expected {expected_folds} OOF files, found {len(paths)}.")
    threshold = select_operating_threshold(
        oof["label"].to_numpy(),
        oof["probability"].to_numpy(),
        cfg["evaluation"].get("threshold_strategy", "youden"),
        target_sensitivity=float(
            cfg["evaluation"].get("target_sensitivity", 0.95)
        ),
    )
    _report(oof, threshold, cfg, output_dir, "oof")
    with open(output_dir / "operating_threshold.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": threshold,
                "strategy": cfg["evaluation"].get(
                    "threshold_strategy", "youden"
                ),
                "target_sensitivity": cfg["evaluation"].get(
                    "target_sensitivity"
                ),
            },
            handle,
            indent=2,
        )
    return threshold


def ensemble_test(checkpoint_dir: Path, output_dir: Path) -> dict:
    checkpoints = sorted(checkpoint_dir.glob("fold_*/best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found under {checkpoint_dir}")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    expected_folds = int(
        first["config"]["data"]["cross_validation"].get("n_splits", 5)
    )
    if len(checkpoints) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} fold checkpoints, found {len(checkpoints)}."
        )
    fold_predictions = [predict_checkpoint(path, "test") for path in checkpoints]
    reference = fold_predictions[0][["patient_id", "label"]].sort_values("patient_id")
    probability_columns = []
    for predictions in fold_predictions:
        aligned = predictions.sort_values("patient_id")
        if not reference.reset_index(drop=True).equals(
            aligned[["patient_id", "label"]].reset_index(drop=True)
        ):
            raise ValueError("Fold checkpoints do not contain the same test patients.")
        probability_columns.append(aligned["probability"].to_numpy())
    ensemble = reference.copy()
    ensemble["probability"] = np.mean(probability_columns, axis=0)
    attention_columns = [
        column
        for column in fold_predictions[0].columns
        if column.startswith("attention_")
    ]
    for column in attention_columns:
        ensemble[column] = np.mean(
            [
                predictions.sort_values("patient_id")[column].to_numpy()
                for predictions in fold_predictions
            ],
            axis=0,
        )
    threshold_file = output_dir / "operating_threshold.json"
    if threshold_file.exists():
        threshold = json.loads(threshold_file.read_text())["threshold"]
    else:
        threshold = aggregate_oof(checkpoint_dir, output_dir)
    cfg = first["config"]
    return _report(ensemble, float(threshold), cfg, output_dir, "test_ensemble")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir)
    if args.aggregate_oof or args.ensemble_test:
        if not args.checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required for aggregate/ensemble modes."
            )
        checkpoint_dir = Path(args.checkpoint_dir)
        if args.aggregate_oof:
            aggregate_oof(checkpoint_dir, output_dir)
        if args.ensemble_test:
            ensemble_test(checkpoint_dir, output_dir)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for single-checkpoint evaluation.")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictions = predict_checkpoint(checkpoint_path, args.split)
    _report(
        predictions,
        float(checkpoint.get("threshold", 0.5)),
        checkpoint["config"],
        output_dir,
        f"{args.split}_fold_{checkpoint['split_metadata']['fold']}",
    )


if __name__ == "__main__":
    main()
