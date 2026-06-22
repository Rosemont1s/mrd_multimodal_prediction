"""Data module for MRD Multimodal Prediction."""

from src.data.clinical_builder import (
    MODEL_FEATURE_COLUMNS,
    attach_mrd_labels,
    build_baseline_clinical_table,
)
from src.data.cohort_audit import (
    AuditResult,
    audit_and_build_cohort,
    write_table_templates,
)
from src.data.ct_manifest import load_ct_path_map
from src.data.clinical_processor import ClinicalProcessor
from src.data.manifest import create_split_manifest, load_split_manifest

__all__ = [
    "ClinicalProcessor",
    "MODEL_FEATURE_COLUMNS",
    "build_baseline_clinical_table",
    "attach_mrd_labels",
    "AuditResult",
    "audit_and_build_cohort",
    "write_table_templates",
    "load_ct_path_map",
    "MRDMultimodalDataset",
    "DataBundle",
    "build_data_bundle",
    "get_dataloaders",
    "create_split_manifest",
    "load_split_manifest",
    "get_train_transforms",
    "get_val_transforms",
    "resolve_clinical_feature_columns",
]


def __getattr__(name: str):
    """Load torch/MONAI-dependent data objects only when requested."""
    if name in {
        "DataBundle",
        "MRDMultimodalDataset",
        "build_data_bundle",
        "get_dataloaders",
        "resolve_clinical_feature_columns",
    }:
        from src.data import dataset

        return getattr(dataset, name)
    if name in {"get_train_transforms", "get_val_transforms"}:
        from src.data import transforms

        return getattr(transforms, name)
    raise AttributeError(name)
