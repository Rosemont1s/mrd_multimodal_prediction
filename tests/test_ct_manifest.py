from pathlib import Path

import pandas as pd
import pytest

from src.data.ct_manifest import load_ct_path_map


def test_ct_manifest_maps_prespecified_phases(tmp_path):
    phases = ["nc", "arterial", "portal", "delayed"]
    rows = []
    for phase in phases:
        path = tmp_path / f"{phase}.nii.gz"
        path.touch()
        rows.append(
            {
                "patient_id": "p1",
                "phase_name": phase,
                "image_path": str(path),
            }
        )
    manifest = tmp_path / "ct_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    data_cfg = {
        "ct_sequences": phases,
        "cohort_tables": {"ct_manifest": str(manifest)},
    }

    paths = load_ct_path_map(data_cfg, ["p1"])

    assert list(paths["p1"]) == phases
    assert paths["p1"]["portal"] == tmp_path / "portal.nii.gz"


def test_ct_manifest_rejects_duplicate_patient_phase(tmp_path):
    manifest = tmp_path / "ct_manifest.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "phase_name": ["portal", "portal"],
            "image_path": ["a.nii.gz", "b.nii.gz"],
        }
    ).to_csv(manifest, index=False)
    data_cfg = {
        "ct_sequences": ["portal"],
        "cohort_tables": {"ct_manifest": str(manifest)},
    }

    with pytest.raises(ValueError, match="duplicate"):
        load_ct_path_map(data_cfg)
