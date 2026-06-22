"""Datasets and fold-aware data bundle for multimodal MRD prediction."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.clinical_processor import ClinicalProcessor
from src.data.ct_manifest import load_ct_path_map
from src.data.manifest import load_split_manifest
from src.data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def resolve_clinical_feature_columns(data_cfg: Dict[str, Any]) -> List[str] | None:
    """Resolve the active baseline profile to an explicit feature allowlist."""
    profile = data_cfg.get("active_clinical_profile")
    profiles = data_cfg.get("clinical_feature_profiles", {})
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown clinical feature profile: {profile}")
        return list(profiles[profile])
    configured = data_cfg.get("clinical_feature_columns")
    return list(configured) if configured is not None else None


def _find_ct_path(patient_dir: Path, sequence: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = patient_dir / f"{sequence}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing sequence '{sequence}' under {patient_dir}")


def _augment_cached_ct(ct: torch.Tensor) -> torch.Tensor:
    """Apply synchronized spatial augmentation to a cached (C,D,H,W) tensor."""
    if random.random() < 0.5:
        ct = torch.flip(ct, dims=(1,))
    if random.random() < 0.5:
        ct = torch.flip(ct, dims=(2,))
    if random.random() < 0.5:
        ct = torch.flip(ct, dims=(3,))
    if random.random() < 0.5:
        ct = torch.rot90(ct, random.randint(1, 3), dims=(2, 3))
    if random.random() < 0.3:
        ct = ct + torch.randn_like(ct) * 0.05
    return ct


class MRDMultimodalDataset(Dataset):
    """Four registered CT sequences, fitted clinical features, and one label."""

    def __init__(
        self,
        patient_ids: Sequence[str],
        cfg: Dict[str, Any],
        clinical_processor: ClinicalProcessor,
        training: bool = False,
    ) -> None:
        self.patient_ids = [str(patient_id) for patient_id in patient_ids]
        self.cfg = cfg
        self.clinical_processor = clinical_processor
        self.training = training
        self.variant = cfg.get("model", {}).get("variant", "gated_fusion")
        data_cfg = cfg["data"]
        self.raw_dir = Path(data_cfg["raw_dir"])
        self.sequences = list(data_cfg["ct_sequences"])
        self.ct_path_map = (
            load_ct_path_map(data_cfg, self.patient_ids)
            if data_cfg.get("use_ct_manifest", True)
            and self.variant != "clinical_only"
            else None
        )
        self.use_cache = bool(data_cfg.get("use_cache", False))
        self.cache_dir = Path(data_cfg.get("cache_dir", "data/processed/ct_cache"))
        self.keys = [f"image_{index}" for index in range(len(self.sequences))]
        transform_cfg = cfg["ct_preprocessing"]
        self.transform = (
            get_train_transforms(transform_cfg, self.keys)
            if training
            else get_val_transforms(transform_cfg, self.keys)
        )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_ct(self, patient_id: str) -> torch.Tensor:
        cache_path = self.cache_dir / f"{patient_id}_ct.pt"
        if self.use_cache:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cached CT not found: {cache_path}. Run preprocessing with "
                    "--cache-ct or disable data.use_cache."
                )
            ct = torch.load(cache_path, map_location="cpu", weights_only=True).float()
            return _augment_cached_ct(ct) if self.training else ct

        if self.ct_path_map is not None:
            sample = {
                key: str(self.ct_path_map[patient_id][sequence])
                for key, sequence in zip(self.keys, self.sequences)
            }
        else:
            patient_dir = self.raw_dir / patient_id
            sample = {
                key: str(_find_ct_path(patient_dir, sequence))
                for key, sequence in zip(self.keys, self.sequences)
            }
        transformed = self.transform(sample)
        volumes = [torch.as_tensor(transformed[key]).float() for key in self.keys]
        return torch.cat(volumes, dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[index]
        clinical = (
            torch.empty(0, dtype=torch.float32)
            if self.variant == "ct_only"
            else torch.from_numpy(
                self.clinical_processor.transform(patient_id)
            ).float()
        )
        label = torch.tensor(
            [self.clinical_processor.get_label(patient_id)], dtype=torch.float32
        )
        return {
            "ct": (
                torch.empty(0, dtype=torch.float32)
                if self.variant == "clinical_only"
                else self._load_ct(patient_id)
            ),
            "clinical": clinical,
            "label": label,
            "patient_id": patient_id,
        }


@dataclass
class DataBundle:
    """Fold-specific data objects and leakage-safe preprocessing state."""

    loaders: Dict[str, DataLoader]
    clinical_input_dim: int
    clinical_processor: ClinicalProcessor
    split_metadata: Dict[str, Any]
    train_positive_weight: float


def build_data_bundle(
    cfg: Dict[str, Any],
    fold: int,
    include_test: bool = True,
) -> DataBundle:
    """Build one fold with a processor fitted only on that fold's training IDs."""
    manifest = load_split_manifest(cfg)
    n_splits = int(cfg["data"]["cross_validation"].get("n_splits", 5))
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], got {fold}")

    cv_rows = manifest[manifest["split"] == "cv"]
    train_rows = cv_rows[cv_rows["fold"] != fold]
    val_rows = cv_rows[cv_rows["fold"] == fold]
    test_rows = manifest[manifest["split"] == "test"]
    split_sets = {
        "train": set(train_rows["patient_id"]),
        "val": set(val_rows["patient_id"]),
        "test": set(test_rows["patient_id"]),
    }
    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError("Patient leakage detected between train and validation.")
    if (split_sets["train"] | split_sets["val"]) & split_sets["test"]:
        raise RuntimeError("Patient leakage detected between CV and test splits.")

    data_cfg = cfg["data"]
    processor = ClinicalProcessor(
        data_cfg["clinical_csv"],
        data_cfg["patient_id_column"],
        data_cfg["label_column"],
        feature_columns=resolve_clinical_feature_columns(data_cfg),
        forbidden_feature_columns=data_cfg.get("forbidden_clinical_columns"),
    ).fit(train_rows["patient_id"].tolist())

    datasets = {
        "train": MRDMultimodalDataset(
            train_rows["patient_id"].tolist(), cfg, processor, training=True
        ),
        "val": MRDMultimodalDataset(
            val_rows["patient_id"].tolist(), cfg, processor, training=False
        ),
    }
    if include_test and not test_rows.empty:
        datasets["test"] = MRDMultimodalDataset(
            test_rows["patient_id"].tolist(), cfg, processor, training=False
        )

    train_cfg = cfg["training"]
    batch_size = int(train_cfg.get("batch_size", 4))
    loader_kwargs = {
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, **loader_kwargs
        ),
    }
    if "test" in datasets:
        loaders["test"] = DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, **loader_kwargs
        )

    positives = int(train_rows["label"].sum())
    negatives = len(train_rows) - positives
    if positives == 0:
        raise ValueError(f"Fold {fold} training split contains no positive patients.")

    return DataBundle(
        loaders=loaders,
        clinical_input_dim=processor.get_feature_dim(),
        clinical_processor=processor,
        split_metadata={
            "fold": fold,
            "train_ids": train_rows["patient_id"].tolist(),
            "val_ids": val_rows["patient_id"].tolist(),
            "test_ids": test_rows["patient_id"].tolist(),
            "manifest": manifest.to_dict(orient="records"),
        },
        train_positive_weight=float(negatives / positives),
    )


def get_dataloaders(cfg: Dict[str, Any], fold: int = 0) -> Dict[str, DataLoader]:
    """Backward-compatible loader factory."""
    return build_data_bundle(cfg, fold).loaders
