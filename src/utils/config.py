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
    if len(sequences) != 4 or len(set(sequences)) != 4:
        raise ValueError("data.ct_sequences must contain four unique sequence names.")

    cv_cfg = data_cfg.get("cross_validation", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    test_size = float(cv_cfg.get("test_size", 0.15))
    if n_splits < 2:
        raise ValueError("data.cross_validation.n_splits must be at least 2.")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("data.cross_validation.test_size must be in [0, 1).")

    variant = cfg.get("model", {}).get("variant", "gated_fusion")
    if variant not in {"gated_fusion", "clinical_only", "ct_only"}:
        raise ValueError(
            "model.variant must be gated_fusion, clinical_only, or ct_only."
        )

    if cfg["ct_extractor"].get("backbone", "resnet18") != "resnet18":
        raise ValueError("The completed pipeline currently supports resnet18 only.")

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
