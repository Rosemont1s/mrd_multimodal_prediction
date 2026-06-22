import numpy as np
import pandas as pd

from src.data.clinical_processor import ClinicalProcessor


def test_processor_fits_training_categories_only(tmp_path):
    path = tmp_path / "clinical.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "mrd_status": [0, 1, 0],
            "age": [50.0, 70.0, 60.0],
            "group": ["A", "B", "UNSEEN"],
        }
    ).to_csv(path, index=False)
    processor = ClinicalProcessor(path, "patient_id", "mrd_status").fit(["p1", "p2"])
    transformed = processor.transform("p3")
    assert transformed.dtype == np.float32
    assert transformed.shape == (processor.get_feature_dim(),)
    assert processor._encoder.categories_[0].tolist() == ["A", "B"]


def test_processor_uses_only_allowlisted_features(tmp_path):
    path = tmp_path / "clinical.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "mrd_status": [0, 1],
            "age": [50.0, 70.0],
            "sex": ["F", "M"],
            "recurrence": [0, 1],
        }
    ).to_csv(path, index=False)

    processor = ClinicalProcessor(
        path,
        "patient_id",
        "mrd_status",
        feature_columns=["age", "sex"],
        forbidden_feature_columns=["recurrence"],
    ).fit(["p1", "p2"])

    assert processor.numeric_cols == ["age"]
    assert processor.categorical_cols == ["sex"]


def test_processor_rejects_forbidden_allowlisted_feature(tmp_path):
    path = tmp_path / "clinical.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "mrd_status": [0, 1],
            "recurrence": [0, 1],
        }
    ).to_csv(path, index=False)

    with np.testing.assert_raises_regex(ValueError, "forbidden"):
        ClinicalProcessor(
            path,
            "patient_id",
            "mrd_status",
            feature_columns=["recurrence"],
            forbidden_feature_columns=["recurrence"],
        )
