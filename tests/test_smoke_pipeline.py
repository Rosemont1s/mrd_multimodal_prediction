from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from scripts.evaluate import predict_checkpoint
from scripts.train import build_optimizer, build_scheduler
from src.data.dataset import build_data_bundle
from src.data.manifest import create_split_manifest
from src.models.mrd_predictor import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer


class _NullLogger:
    def log_scalars(self, *_args, **_kwargs):
        pass


def test_synthetic_cpu_training_and_evaluation(tmp_path):
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    patients = [f"p{index:02d}" for index in range(20)]
    rows = []
    for index, patient_id in enumerate(patients):
        patient_dir = raw_dir / patient_id
        patient_dir.mkdir()
        for sequence in config["data"]["ct_sequences"]:
            image = nib.Nifti1Image(
                np.full((12, 12, 12), index, dtype=np.float32), np.eye(4)
            )
            nib.save(image, patient_dir / f"{sequence}.nii.gz")
        rows.append(
            {
                "patient_id": patient_id,
                "mrd_status": index % 2,
                "age": 40 + index,
                "sex": "F" if index % 2 else "M",
            }
        )
    clinical_path = raw_dir / "clinical_data.csv"
    pd.DataFrame(rows).to_csv(clinical_path, index=False)

    config["data"].update(
        {
            "raw_dir": str(raw_dir),
            "clinical_csv": str(clinical_path),
            "manifest_path": str(tmp_path / "splits.csv"),
            "num_workers": 0,
            "pin_memory": False,
            "clinical_feature_columns": ["age", "sex"],
            "clinical_feature_profiles": {
                "clinical_pathology": ["age", "sex"],
                "stage_only": ["age"],
            },
            "active_clinical_profile": "clinical_pathology",
            "require_readiness_report": False,
            "use_ct_manifest": False,
        }
    )
    config["data"]["cross_validation"] = {
        "n_splits": 2,
        "strategy": "random_holdout",
        "test_size": 0.2,
    }
    config["ct_preprocessing"].update(
        {"spatial_size": [16, 16, 16], "target_spacing": [1.0, 1.0, 1.0]}
    )
    config["model"]["variant"] = "clinical_only"
    config["training"].update(
        {
            "num_epochs": 1,
            "batch_size": 4,
            "stage2_start_epoch": 10,
            "scheduler": "none",
        }
    )
    config["evaluation"]["bootstrap_samples"] = 10
    config["device"]["accelerator"] = "cpu"
    create_split_manifest(config, validate_images=False)

    bundle = build_data_bundle(config, fold=0)
    model = build_model(config, bundle.clinical_input_dim)
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model=model,
        train_loader=bundle.loaders["train"],
        val_loader=bundle.loaders["val"],
        criterion=build_loss(config, bundle.train_positive_weight),
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, config),
        cfg=config,
        exp_logger=_NullLogger(),
        clinical_processor=bundle.clinical_processor,
        split_metadata=bundle.split_metadata,
        output_dir=str(tmp_path / "fold_0"),
    )
    result = trainer.fit()
    predictions = predict_checkpoint(Path(result["best_checkpoint"]), "val")
    assert len(predictions) == len(bundle.split_metadata["val_ids"])
    assert predictions["probability"].between(0, 1).all()
