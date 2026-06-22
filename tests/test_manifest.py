from pathlib import Path

import pandas as pd
import yaml

from src.data.manifest import create_split_manifest


def test_split_manifest_is_reproducible_and_disjoint(tmp_path):
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    patients = [f"p{index:02d}" for index in range(20)]
    for patient_id in patients:
        patient_dir = raw_dir / patient_id
        patient_dir.mkdir()
        for sequence in config["data"]["ct_sequences"]:
            (patient_dir / f"{sequence}.nii.gz").touch()
    clinical_path = raw_dir / "clinical_data.csv"
    pd.DataFrame(
        {
            "patient_id": patients,
            "mrd_status": [index % 2 for index in range(20)],
            "age": range(20),
        }
    ).to_csv(clinical_path, index=False)

    config["data"].update(
        {
            "raw_dir": str(raw_dir),
            "clinical_csv": str(clinical_path),
            "manifest_path": str(tmp_path / "splits.csv"),
        }
    )
    config["data"]["cross_validation"] = {"n_splits": 2, "test_size": 0.2}
    first = create_split_manifest(config, validate_images=False)
    second = create_split_manifest(config, validate_images=False)
    pd.testing.assert_frame_equal(first, second)
    assert set(first[first["split"] == "cv"]["patient_id"]).isdisjoint(
        set(first[first["split"] == "test"]["patient_id"])
    )
    assert set(first[first["split"] == "cv"]["fold"]) == {0, 1}
