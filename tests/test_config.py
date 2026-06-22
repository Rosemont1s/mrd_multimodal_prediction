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
    with pytest.raises(ValueError, match="four unique"):
        validate_config(config)

