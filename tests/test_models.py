import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.models.mrd_predictor import build_model


@pytest.fixture
def config():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg["ct_extractor"]["pretrained"] = False
    return cfg


@pytest.mark.parametrize("variant", ["gated_fusion", "clinical_only", "ct_only"])
def test_model_variants_return_binary_logits(config, variant):
    pytest.importorskip("monai")
    cfg = copy.deepcopy(config)
    cfg["model"]["variant"] = variant
    model = build_model(cfg, clinical_input_dim=7)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.randn(1, 4, 32, 32, 32),
            torch.randn(1, 7),
        )
    assert output["logits"].shape == (1, 1)
    assert output["probs"].shape == (1, 1)


def test_clinical_only_state_dict_round_trip(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["model"]["variant"] = "clinical_only"
    model = build_model(cfg, clinical_input_dim=7)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(cfg, clinical_input_dim=7)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
