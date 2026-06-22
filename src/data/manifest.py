"""Patient-level cohort validation and reproducible split manifests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.data.ct_manifest import load_ct_path_map

logger = logging.getLogger(__name__)


def _ct_path(patient_dir: Path, sequence: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = patient_dir / f"{sequence}{suffix}"
        if candidate.exists():
            return candidate
    return patient_dir / f"{sequence}.nii.gz"


def validate_patient_files(
    patient_ids: Iterable[str],
    raw_dir: str | Path,
    sequences: List[str],
    check_geometry: bool = True,
    path_map: Dict[str, Dict[str, Path]] | None = None,
) -> None:
    """Validate CT completeness and registered geometry for every patient."""
    raw_dir = Path(raw_dir)
    errors: List[str] = []
    for patient_id in patient_ids:
        if path_map is None:
            paths = [_ct_path(raw_dir / str(patient_id), seq) for seq in sequences]
        else:
            patient_paths = path_map.get(str(patient_id), {})
            paths = [
                patient_paths.get(
                    seq, raw_dir / str(patient_id) / f"{seq}.nii.gz"
                )
                for seq in sequences
            ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            errors.append(f"{patient_id}: missing {missing}")
            continue
        if not check_geometry:
            continue

        images = [nib.load(str(path)) for path in paths]
        reference = images[0]
        for sequence, image in zip(sequences[1:], images[1:]):
            if image.shape != reference.shape:
                errors.append(
                    f"{patient_id}/{sequence}: shape {image.shape} != "
                    f"{reference.shape}"
                )
            if not np.allclose(image.affine, reference.affine, atol=1e-3):
                errors.append(f"{patient_id}/{sequence}: affine mismatch")
        for sequence, image in zip(sequences, images):
            data = np.asanyarray(image.dataobj)
            if not np.isfinite(data).all():
                errors.append(f"{patient_id}/{sequence}: non-finite voxels")

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"CT cohort validation failed:\n{preview}")


def create_split_manifest(
    cfg: Dict[str, Any], validate_images: bool = True
) -> pd.DataFrame:
    """Create patient-level development folds and a frozen test cohort."""
    data_cfg = cfg["data"]
    if data_cfg.get("require_readiness_report", True):
        report_path = Path(data_cfg["readiness_report"])
        if not report_path.exists():
            raise FileNotFoundError(
                f"Readiness report not found: {report_path}. "
                "Run scripts/audit_dataset.py first."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("ready_for_definitive_training", False):
            raise ValueError(
                "Dataset readiness report contains unresolved blockers."
            )
    csv_path = Path(data_cfg["clinical_csv"])
    raw_dir = Path(data_cfg["raw_dir"])
    id_col = data_cfg["patient_id_column"]
    label_col = data_cfg["label_column"]
    cv_cfg = data_cfg.get("cross_validation", {})
    seed = int(data_cfg.get("random_seed", 42))
    n_splits = int(cv_cfg.get("n_splits", 5))
    test_size = float(cv_cfg.get("test_size", 0.15))
    strategy = cv_cfg.get("strategy", "random_holdout")

    clinical = pd.read_csv(csv_path)
    if id_col not in clinical or label_col not in clinical:
        raise ValueError(f"Clinical CSV must contain '{id_col}' and '{label_col}'.")
    if clinical[id_col].duplicated().any():
        duplicates = clinical.loc[clinical[id_col].duplicated(), id_col].tolist()
        raise ValueError(f"Duplicate patient IDs in clinical CSV: {duplicates[:10]}")

    clinical[id_col] = clinical[id_col].astype(str)
    labels = pd.to_numeric(clinical[label_col], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"'{label_col}' must contain only non-null binary labels.")
    clinical[label_col] = labels.astype(int)

    use_manifest = bool(data_cfg.get("use_ct_manifest", True))
    path_map = (
        load_ct_path_map(data_cfg, clinical[id_col].tolist())
        if use_manifest
        else None
    )
    available = sorted(
        patient_id
        for patient_id in clinical[id_col]
        if (
            patient_id in path_map
            if path_map is not None
            else (raw_dir / patient_id).is_dir()
        )
    )
    if len(available) != len(clinical):
        missing = sorted(set(clinical[id_col]) - set(available))
        raise ValueError(
            f"Clinical patients without complete CT inputs: {missing[:20]}"
        )

    cohort_columns = [label_col]
    if strategy == "temporal_cohort":
        cohort_column = cv_cfg.get("cohort_column", "cohort_period")
        if cohort_column not in clinical:
            raise ValueError(
                f"Temporal splitting requires clinical column '{cohort_column}'."
            )
        cohort_columns.append(cohort_column)
    cohort = clinical.set_index(id_col).loc[available, cohort_columns].reset_index()
    validate_patient_files(
        cohort[id_col],
        raw_dir,
        list(data_cfg["ct_sequences"]),
        check_geometry=validate_images,
        path_map=path_map,
    )

    indices = np.arange(len(cohort))
    y = cohort[label_col].astype(int).to_numpy()
    if strategy == "temporal_cohort":
        cohort_column = cv_cfg.get("cohort_column", "cohort_period")
        values = cohort[cohort_column].astype(str).str.lower()
        development_value = str(
            cv_cfg.get("development_value", "retrospective")
        ).lower()
        test_value = str(cv_cfg.get("test_value", "prospective")).lower()
        unknown = sorted(
            set(values) - {development_value, test_value}
        )
        if unknown:
            raise ValueError(
                f"Unexpected values in '{cohort_column}': {unknown}"
            )
        cv_idx = indices[values.eq(development_value)]
        test_idx = indices[values.eq(test_value)]
        if len(cv_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                "Temporal splitting requires non-empty development and test cohorts."
            )
    elif strategy == "random_holdout" and test_size > 0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        cv_idx, test_idx = next(splitter.split(indices, y))
    elif strategy == "random_holdout":
        cv_idx, test_idx = indices, np.array([], dtype=int)
    else:
        raise ValueError(
            "data.cross_validation.strategy must be temporal_cohort "
            "or random_holdout."
        )

    cv_labels = y[cv_idx]
    if np.min(np.bincount(cv_labels)) < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} non-test patients for "
            f"{n_splits}-fold stratification."
        )

    manifest = cohort.rename(
        columns={id_col: "patient_id", label_col: "label"}
    )
    manifest["split"] = "cv"
    manifest["fold"] = -1
    manifest.loc[test_idx, "split"] = "test"

    fold_splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for fold, (_, validation_rel_idx) in enumerate(
        fold_splitter.split(cv_idx, cv_labels)
    ):
        manifest.loc[cv_idx[validation_rel_idx], "fold"] = fold

    if (manifest.loc[manifest["split"] == "cv", "fold"] < 0).any():
        raise RuntimeError("Some cross-validation patients were not assigned a fold.")
    if len(test_idx) and set(
        manifest.loc[manifest["split"] == "test", "label"]
    ) != {0, 1}:
        raise ValueError("The test holdout must contain both outcome classes.")

    output = Path(data_cfg["manifest_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.sort_values("patient_id").to_csv(output, index=False)
    logger.info("Saved split manifest with %d patients to %s", len(manifest), output)
    return manifest


def load_split_manifest(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load and validate an existing split manifest."""
    path = Path(cfg["data"]["manifest_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run scripts/preprocess.py first."
        )
    manifest = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "label", "split", "fold"}
    if not required.issubset(manifest.columns):
        missing = sorted(required - set(manifest))
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate patient IDs.")
    if not set(manifest["split"]).issubset({"cv", "test"}):
        raise ValueError("Manifest split values must be 'cv' or 'test'.")
    return manifest
