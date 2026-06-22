"""Model-input path handling for the canonical CT series manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def load_ct_path_map(
    data_cfg: Dict[str, Any],
    patient_ids: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Path]]:
    """Return patient/phase paths from the linked CT manifest."""
    manifest_path = Path(data_cfg["cohort_tables"]["ct_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"CT manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {"patient_id", "phase_name", "image_path"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"CT manifest is missing columns: {missing}")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["phase_name"] = frame["phase_name"].astype(str).str.strip()
    if patient_ids is not None:
        selected = {str(patient_id) for patient_id in patient_ids}
        frame = frame[frame["patient_id"].isin(selected)]
    if frame.duplicated(["patient_id", "phase_name"]).any():
        raise ValueError("CT manifest contains duplicate patient/phase rows.")
    expected = list(data_cfg["ct_sequences"])
    result: Dict[str, Dict[str, Path]] = {}
    for patient_id, group in frame.groupby("patient_id"):
        phases = dict(
            zip(group["phase_name"], group["image_path"].map(Path))
        )
        missing = [phase for phase in expected if phase not in phases]
        if missing:
            raise ValueError(
                f"Patient {patient_id} is missing CT phases: {missing}"
            )
        result[patient_id] = {phase: phases[phase] for phase in expected}
    return result
