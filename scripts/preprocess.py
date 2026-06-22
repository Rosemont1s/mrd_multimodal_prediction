#!/usr/bin/env python3
"""Validate the cohort, create split manifests, and optionally cache CT tensors."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.manifest import create_split_manifest
from src.data.ct_manifest import load_ct_path_map
from src.data.transforms import get_val_transforms
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-ct", action="store_true")
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip expensive geometry/finite-value checks (not recommended).",
    )
    return parser.parse_args()


def cache_ct_data(cfg: dict, patient_ids: list[str]) -> None:
    data_cfg = cfg["data"]
    sequences = list(data_cfg["ct_sequences"])
    keys = [f"image_{index}" for index in range(len(sequences))]
    transform = get_val_transforms(cfg["ct_preprocessing"], keys)
    raw_dir = Path(data_cfg["raw_dir"])
    path_map = (
        load_ct_path_map(data_cfg, patient_ids)
        if data_cfg.get("use_ct_manifest", True)
        else None
    )
    cache_dir = Path(data_cfg.get("cache_dir", "data/processed/ct_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    for patient_id in tqdm(patient_ids, desc="Caching deterministic CT"):
        sample = {}
        for key, sequence in zip(keys, sequences):
            if path_map is not None:
                path = path_map[patient_id][sequence]
            else:
                candidates = [
                    raw_dir / patient_id / f"{sequence}.nii.gz",
                    raw_dir / patient_id / f"{sequence}.nii",
                ]
                path = next(
                    (candidate for candidate in candidates if candidate.exists()),
                    None,
                )
            if path is None:
                raise FileNotFoundError(
                    f"Missing CT sequence {sequence} for patient {patient_id}"
                )
            sample[key] = str(path)
        output = transform(sample)
        tensor = torch.cat(
            [torch.as_tensor(output[key]).float() for key in keys], dim=0
        )
        torch.save(tensor, cache_dir / f"{patient_id}_ct.pt")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    manifest = create_split_manifest(
        cfg, validate_images=not args.skip_image_validation
    )
    logger.info(
        "Manifest summary:\n%s",
        manifest.groupby(["split", "label"]).size().to_string(),
    )
    if args.cache_ct:
        cache_ct_data(cfg, manifest["patient_id"].tolist())
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
