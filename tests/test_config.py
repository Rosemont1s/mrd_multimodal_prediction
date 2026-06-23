from pathlib import Path

import pytest
import yaml

from src.utils.config import validate_config


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
