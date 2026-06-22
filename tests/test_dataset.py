import torch

from src.data import dataset


def test_cached_spatial_augmentation_is_shared_across_channels(monkeypatch):
    values = iter([0.1, 0.9, 0.9, 0.1, 0.9])
    monkeypatch.setattr(dataset.random, "random", lambda: next(values))
    monkeypatch.setattr(dataset.random, "randint", lambda _a, _b: 1)
    base = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    ct = torch.cat([base + channel * 100 for channel in range(4)], dim=0)
    augmented = dataset._augment_cached_ct(ct)
    differences = augmented - augmented[0:1]
    for channel in range(4):
        assert torch.allclose(
            differences[channel], torch.full_like(differences[channel], channel * 100)
        )

