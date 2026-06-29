"""
Configuration Utilities
========================
YAML configuration loading, deep-merging, and saving utilities.
"""

import copy
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


ConfigDict = Dict[str, Any]

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

REQUIRED_DATA_KEYS = {
    "raw_dir",
    "clinical_csv",
    "patient_id_column",
    "label_column",
}
REQUIRED_COHORT_TABLES = {
    "cohort",
    "preoperative",
    "pathology",
    "mrd",
    "ct_manifest",
}
REQUIRED_TEMPORAL_SPLIT_KEYS = {
    "cohort_column",
    "development_value",
    "test_value",
}
REQUIRED_LAB_UNITS = {"cea", "ca199"}

VALID_SPLIT_STRATEGIES = {"random_holdout", "temporal_cohort"}
VALID_THRESHOLD_STRATEGIES = {"youden", "target_sensitivity"}
VALID_MODEL_VARIANTS = {"gated_fusion", "clinical_only", "ct_only"}
VALID_PHASE_FUSIONS = {"attention", "mean"}
VALID_SPATIAL_STRATEGIES = {"resize", "pad_crop"}
VALID_FUSION_METHODS = {"gated", "concat"}
TRUE_STRINGS = {"true", "yes"}
FALSE_STRINGS = {"false", "no"}
NONE_STRINGS = {"null", "none"}


def _missing_keys(mapping: Mapping[str, Any], required: set[str]) -> list[str]:
    """Return required keys absent from a mapping in deterministic order."""
    return sorted(required - set(mapping))


def _require_mapping(mapping: Mapping[str, Any], key: str, path: str) -> ConfigDict:
    """Fetch a nested mapping and fail with a schema-oriented error."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        full_path = f"{path}.{key}" if path else key
        raise ValueError(f"{full_path} must be a mapping.")
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    path: str,
) -> None:
    """Validate that a mapping contains all required schema keys."""
    missing = _missing_keys(mapping, required)
    if missing:
        raise ValueError(f"Missing {path} configuration keys: {missing}")


def _validate_string_list(value: Any, path: str) -> list[str]:
    """Validate a non-empty list of string values and return it."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list.")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{path} must contain non-empty strings.")
    return value


def load_config(config_path: str | Path = "configs/default.yaml") -> ConfigDict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    validate_config(cfg)
    logger.info("Loaded configuration from: %s", path)
    return cfg


def validate_config(cfg: ConfigDict) -> None:
    """Validate the canonical nested configuration schema."""
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")

    # Fail fast on structure before reading nested values; this makes CLI
    # override mistakes much easier to diagnose.
    missing = _missing_keys(cfg, REQUIRED_SECTIONS)
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    for section in sorted(REQUIRED_SECTIONS):
        _require_mapping(cfg, section, "")

    data_cfg = cfg["data"]
    _require_keys(data_cfg, REQUIRED_DATA_KEYS, "data")

    sequences = data_cfg.get("ct_sequences", [])
    if (
        not isinstance(sequences, list)
        or not sequences
        or any(not isinstance(sequence, str) or not sequence for sequence in sequences)
        or len(sequences) != len(set(sequences))
    ):
        raise ValueError(
            "data.ct_sequences must contain one or more unique CT phase names."
        )

    cv_cfg = data_cfg.get("cross_validation", {})
    if not isinstance(cv_cfg, dict):
        raise ValueError("data.cross_validation must be a mapping.")
    n_splits = int(cv_cfg.get("n_splits", 5))
    test_size = float(cv_cfg.get("test_size", 0.15))
    split_strategy = cv_cfg.get("strategy", "random_holdout")
    if n_splits < 2:
        raise ValueError("data.cross_validation.n_splits must be at least 2.")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("data.cross_validation.test_size must be in [0, 1).")
    if split_strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(
            "data.cross_validation.strategy must be random_holdout "
            "or temporal_cohort."
        )
    if split_strategy == "temporal_cohort":
        # Temporal holdout is the leakage-sensitive path: require explicit
        # cohort labels rather than silently falling back to random splitting.
        missing_temporal = _missing_keys(cv_cfg, REQUIRED_TEMPORAL_SPLIT_KEYS)
        if missing_temporal:
            raise ValueError(
                "Temporal split configuration is missing: "
                f"{missing_temporal}"
            )

    cohort_tables = data_cfg.get("cohort_tables", {})
    if not isinstance(cohort_tables, dict):
        raise ValueError("data.cohort_tables must be a mapping.")
    missing_tables = _missing_keys(cohort_tables, REQUIRED_COHORT_TABLES)
    if missing_tables:
        raise ValueError(
            f"data.cohort_tables is missing entries: {missing_tables}"
        )
    endpoint_cfg = data_cfg.get("mrd_endpoint", {})
    if not isinstance(endpoint_cfg, dict):
        raise ValueError("data.mrd_endpoint must be a mapping.")
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
    if (
        not isinstance(expected_units, dict)
        or not REQUIRED_LAB_UNITS.issubset(expected_units)
    ):
        raise ValueError(
            "data.expected_lab_units must define CEA and CA19-9 units."
        )

    feature_columns = data_cfg.get("clinical_feature_columns")
    feature_column_set = None
    if feature_columns is not None:
        feature_columns = _validate_string_list(
            feature_columns,
            "data.clinical_feature_columns",
        )
        feature_column_set = set(feature_columns)
        if len(feature_columns) != len(feature_column_set):
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
        columns = _validate_string_list(
            columns,
            f"Clinical feature profile '{name}'",
        )
        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Clinical feature profile '{name}' contains duplicates."
            )
        # Profiles are checked against the master allowlist when one is
        # configured. If no allowlist exists, downstream preprocessing still
        # validates columns against the actual clinical table.
        if feature_column_set is None:
            continue
        unknown = sorted(set(columns) - feature_column_set)
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
    if threshold_strategy not in VALID_THRESHOLD_STRATEGIES:
        raise ValueError(
            "evaluation.threshold_strategy must be youden or target_sensitivity."
        )
    target_sensitivity = float(evaluation_cfg.get("target_sensitivity", 0.95))
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("evaluation.target_sensitivity must be in (0, 1].")

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model must be a mapping.")
    variant = model_cfg.get("variant", "gated_fusion")
    if variant not in VALID_MODEL_VARIANTS:
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
    phase_fusion = cfg["ct_extractor"].get("phase_fusion", "attention")
    if phase_fusion not in VALID_PHASE_FUSIONS:
        raise ValueError(
            "ct_extractor.phase_fusion must be attention or mean."
        )

    # Keep CT preprocessing ranges explicit so comparable experiments cannot
    # accidentally train with an inverted or unsupported spatial transform.
    preprocessing_cfg = cfg["ct_preprocessing"]
    intensity_min = float(preprocessing_cfg.get("intensity_min", -200.0))
    intensity_max = float(preprocessing_cfg.get("intensity_max", 300.0))
    if intensity_min >= intensity_max:
        raise ValueError(
            "ct_preprocessing.intensity_min must be less than intensity_max."
        )
    spatial_strategy = preprocessing_cfg.get("spatial_strategy", "resize")
    if spatial_strategy not in VALID_SPATIAL_STRATEGIES:
        raise ValueError(
            "ct_preprocessing.spatial_strategy must be resize or pad_crop."
        )

    # Fusion and classifier dimensions must agree before the model is built.
    projection_dim = int(cfg["fusion"].get("projection_dim", 128))
    fusion_method = cfg["fusion"].get("method", "gated")
    if fusion_method not in VALID_FUSION_METHODS:
        raise ValueError("fusion.method must be gated or concat.")
    classifier_dim = int(cfg["classifier"].get("input_dim", projection_dim))
    if classifier_dim != projection_dim:
        raise ValueError(
            "classifier.input_dim must equal fusion.projection_dim "
            f"({classifier_dim} != {projection_dim})."
        )

    train_cfg = cfg["training"]
    stage2_epoch = int(train_cfg.get("stage2_start_epoch", 10))
    stage3_epoch = int(train_cfg.get("stage3_start_epoch", 30))
    if not 0 <= stage2_epoch < stage3_epoch:
        raise ValueError(
            "training stages must satisfy 0 <= stage2_start_epoch "
            "< stage3_start_epoch."
        )


def merge_configs(
    base: ConfigDict,
    override: ConfigDict,
) -> ConfigDict:
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


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        cfg: Configuration dictionary to save.
        path: Output file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved configuration to: %s", output_path)


def apply_overrides(
    cfg: ConfigDict,
    overrides: Optional[Sequence[str]] = None,
) -> ConfigDict:
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
            logger.warning("Skipping malformed override (no '='): %s", override)
            continue

        key_path, value_str = override.split("=", 1)
        key_path = key_path.strip()
        keys = [key.strip() for key in key_path.split(".")]
        if not key_path or any(not key for key in keys):
            logger.warning("Skipping malformed override (empty key): %s", override)
            continue
        value = _auto_cast(value_str.strip())

        # Create intermediate sections as needed for dotted CLI overrides.
        current: MutableMapping[str, Any] = cfg
        for key in keys[:-1]:
            child = current.get(key)
            if not isinstance(child, MutableMapping):
                child = {}
                current[key] = child
            current = child

        current[keys[-1]] = value
        logger.info(
            "Override applied: %s = %r (%s)",
            key_path,
            value,
            type(value).__name__,
        )

    return cfg


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string value to the most appropriate Python type.

    Tries in order: ``bool``, ``int``, ``float``, then falls back to ``str``.
    """
    normalized = value.lower()

    # Keep an explicit empty override as an empty string rather than YAML null.
    if value == "":
        return value
    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False
    if normalized in NONE_STRINGS:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
