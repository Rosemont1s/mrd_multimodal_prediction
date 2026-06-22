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

