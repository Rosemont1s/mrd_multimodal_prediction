from pathlib import Path

import pytest
import yaml

from src.utils.config import apply_overrides, validate_config


def test_default_config_is_valid():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    validate_config(config)


def test_config_rejects_non_unique_sequences():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["data"]["ct_sequences"] = ["a", "a", "b", "c"]
    with pytest.raises(ValueError, match="unique CT phase"):
        validate_config(config)


def test_config_rejects_forbidden_clinical_feature():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["data"]["clinical_feature_columns"].append("recurrence")
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(config)


def test_config_rejects_invalid_ct_training_stage_order():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["training"]["stage2_start_epoch"] = 30
    config["training"]["stage3_start_epoch"] = 10
    with pytest.raises(ValueError, match="training stages"):
        validate_config(config)


def test_config_rejects_invalid_phase_fusion():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["ct_extractor"]["phase_fusion"] = "channel_average"
    with pytest.raises(ValueError, match="phase_fusion"):
        validate_config(config)


def test_config_rejects_non_mapping_sections():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["training"] = None
    with pytest.raises(ValueError, match="training must be a mapping"):
        validate_config(config)


def test_config_accepts_profiles_without_master_allowlist():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    del config["data"]["clinical_feature_columns"]
    validate_config(config)


def test_apply_overrides_casts_values_and_creates_nested_sections():
    config = {"training": {"batch_size": 2}}
    result = apply_overrides(
        config,
        [
            "training.batch_size=8",
            "data.use_cache=true",
            "checkpoint.monitor_metric=none",
        ],
    )

    assert result is config
    assert result["training"]["batch_size"] == 8
    assert result["data"]["use_cache"] is True
    assert result["checkpoint"]["monitor_metric"] is None
