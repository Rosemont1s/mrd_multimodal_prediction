"""
Configuration Utilities
========================
YAML configuration loading, deep-merging, and saving utilities.
"""

import copy
import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


REQUIRED_SECTIONS = {
    "data",
    "ct_preprocessing",
    "ct_extractor",
    "clinical_encoder",
    "fusion",
    "classifier",
    "training",
    "evaluation",
    "checkpoint",
    "device",
}


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    validate_config(cfg)
    logger.info(f"Loaded configuration from: {config_path}")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate the canonical nested configuration schema."""
    missing = sorted(REQUIRED_SECTIONS - set(cfg))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    data_cfg = cfg["data"]
    required_data = {"raw_dir", "clinical_csv", "patient_id_column", "label_column"}
    missing_data = sorted(required_data - set(data_cfg))
    if missing_data:
        raise ValueError(f"Missing data configuration keys: {missing_data}")

    sequences = data_cfg.get("ct_sequences", [])
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError(
            "data.ct_sequences must contain one or more unique CT phase names."
        )

    cv_cfg = data_cfg.get("cross_validation", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    test_size = float(cv_cfg.get("test_size", 0.15))
    split_strategy = cv_cfg.get("strategy", "random_holdout")
    if n_splits < 2:
        raise ValueError("data.cross_validation.n_splits must be at least 2.")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("data.cross_validation.test_size must be in [0, 1).")
    if split_strategy not in {"random_holdout", "temporal_cohort"}:
        raise ValueError(
            "data.cross_validation.strategy must be random_holdout "
            "or temporal_cohort."
        )
    if split_strategy == "temporal_cohort":
        required_temporal = {
            "cohort_column",
            "development_value",
            "test_value",
        }
        missing_temporal = sorted(required_temporal - set(cv_cfg))
        if missing_temporal:
            raise ValueError(
                "Temporal split configuration is missing: "
                f"{missing_temporal}"
            )

    cohort_tables = data_cfg.get("cohort_tables", {})
    required_tables = {
        "cohort",
        "preoperative",
        "pathology",
        "mrd",
        "ct_manifest",
    }
    missing_tables = sorted(required_tables - set(cohort_tables))
    if missing_tables:
        raise ValueError(
            f"data.cohort_tables is missing entries: {missing_tables}"
        )
    endpoint_cfg = data_cfg.get("mrd_endpoint", {})
    if "assay_definition_finalized" not in endpoint_cfg:
        raise ValueError(
            "data.mrd_endpoint.assay_definition_finalized is required."
        )
    if "blood_draw_window_finalized" not in endpoint_cfg:
        raise ValueError(
            "data.mrd_endpoint.blood_draw_window_finalized is required."
        )
    if endpoint_cfg.get("blood_draw_window_finalized"):
        minimum = endpoint_cfg.get("blood_draw_min_days")
        maximum = endpoint_cfg.get("blood_draw_max_days")
        if minimum is None or maximum is None:
            raise ValueError(
                "Finalized MRD blood-draw window requires minimum and maximum days."
            )
        if int(minimum) < 0 or int(maximum) < int(minimum):
            raise ValueError("Invalid MRD blood-draw window.")
    expected_units = data_cfg.get("expected_lab_units", {})
    if not {"cea", "ca199"}.issubset(expected_units):
        raise ValueError(
            "data.expected_lab_units must define CEA and CA19-9 units."
        )

    feature_columns = data_cfg.get("clinical_feature_columns")
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError(
                "data.clinical_feature_columns must be a non-empty list."
            )
        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError("data.clinical_feature_columns contains duplicates.")
        protected = {
            data_cfg["patient_id_column"],
            data_cfg["label_column"],
        }
        if protected.intersection(feature_columns):
            raise ValueError(
                "data.clinical_feature_columns cannot include the patient ID "
                "or MRD label."
            )
        forbidden = set(data_cfg.get("forbidden_clinical_columns", []))
        overlap = sorted(forbidden.intersection(feature_columns))
        if overlap:
            raise ValueError(
                "Configured clinical features include forbidden post-decision "
                f"columns: {overlap}"
            )
    profiles = data_cfg.get("clinical_feature_profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            "data.clinical_feature_profiles must define named feature lists."
        )
    for name, columns in profiles.items():
        if not isinstance(columns, list) or not columns:
            raise ValueError(
                f"Clinical feature profile '{name}' must be a non-empty list."
            )
        unknown = sorted(set(columns) - set(feature_columns or []))
        if unknown:
            raise ValueError(
                f"Clinical feature profile '{name}' contains unknown columns: "
                f"{unknown}"
            )
    active_profile = data_cfg.get("active_clinical_profile")
    if active_profile not in profiles:
        raise ValueError(
            "data.active_clinical_profile must name a configured profile."
        )

    evaluation_cfg = cfg["evaluation"]
    threshold_strategy = evaluation_cfg.get("threshold_strategy", "youden")
    if threshold_strategy not in {"youden", "target_sensitivity"}:
        raise ValueError(
            "evaluation.threshold_strategy must be youden or target_sensitivity."
        )
    target_sensitivity = float(evaluation_cfg.get("target_sensitivity", 0.95))
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("evaluation.target_sensitivity must be in (0, 1].")

    variant = cfg.get("model", {}).get("variant", "gated_fusion")
    if variant not in {"gated_fusion", "clinical_only", "ct_only"}:
        raise ValueError(
            "model.variant must be gated_fusion, clinical_only, or ct_only."
        )

    if cfg["ct_extractor"].get("backbone", "resnet18") != "resnet18":
        raise ValueError("The completed pipeline currently supports resnet18 only.")
    in_channels = int(cfg["ct_extractor"].get("in_channels", len(sequences)))
    if in_channels != len(sequences):
        raise ValueError(
            "ct_extractor.in_channels must equal the number of CT sequences."
        )

    projection_dim = int(cfg["fusion"].get("projection_dim", 128))
    classifier_dim = int(cfg["classifier"].get("input_dim", projection_dim))
    if classifier_dim != projection_dim:
        raise ValueError(
            "classifier.input_dim must equal fusion.projection_dim "
            f"({classifier_dim} != {projection_dim})."
        )


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively deep-merge two configuration dictionaries.

    Values from ``override`` take precedence over ``base``.  Nested
    dictionaries are merged recursively; all other types are replaced.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        New merged configuration dictionary (does not modify inputs).
    """
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        cfg: Configuration dictionary to save.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to: {path}")


def apply_overrides(
    cfg: Dict[str, Any],
    overrides: Optional[list] = None,
) -> Dict[str, Any]:
    """Apply command-line key=value overrides to a configuration dict.

    Supports dotted keys for nested access, e.g.,
    ``"training.learning_rate=0.001"`` sets
    ``cfg['training']['learning_rate'] = 0.001``.

    Values are auto-cast to ``int``, ``float``, ``bool``, or kept as ``str``.

    Args:
        cfg: Configuration dictionary to modify (in-place).
        overrides: List of ``"key=value"`` strings.

    Returns:
        Modified configuration dictionary.
    """
    if not overrides:
        return cfg

    for override in overrides:
        if "=" not in override:
            logger.warning(f"Skipping malformed override (no '='): {override}")
            continue

        key_path, value_str = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = _auto_cast(value_str.strip())

        # Navigate to the nested dict
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value
        logger.info(f"Override applied: {key_path} = {value} ({type(value).__name__})")

    return cfg


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string value to the most appropriate Python type.

    Tries in order: ``bool``, ``int``, ``float``, then falls back to ``str``.
    """
    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # None
    if value.lower() in ("null", "none"):
        return None
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value
