"""
3D CT Image Transforms for MRD Multimodal Prediction.

This module provides MONAI-based transform pipelines for 3D CT NIfTI volumes.
Training transforms include data augmentation (random flips, rotations, noise,
intensity shifts), while validation transforms apply only deterministic
preprocessing (resampling, HU clipping, cropping/padding, normalization).

Typical config dict::

    cfg = {
        "spatial_size": [64, 128, 128],
        "target_spacing": [2.0, 1.0, 1.0],
        "intensity_min": -1024,
        "intensity_max": 3071,
        "normalize_mean": 0.5,
        "normalize_std": 0.5,
    }
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)

# ──────────────────────────────────────────────────────────────────────
# Default configuration values
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_CFG: Dict[str, Any] = {
    "spatial_size": [64, 128, 128],
    "target_spacing": [2.0, 1.0, 1.0],
    "intensity_min": -1024,
    "intensity_max": 3071,
    "normalize_mean": 0.5,
    "normalize_std": 0.5,
}

# Key used in MONAI dictionary transforms
IMAGE_KEY = "image"


def _resolve_cfg(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge user-supplied config with defaults, returning a complete config.

    Parameters
    ----------
    cfg : dict or None
        User-supplied configuration.  Missing keys are filled from
        ``_DEFAULT_CFG``.

    Returns
    -------
    dict
        Fully-populated configuration dictionary.
    """
    resolved = dict(_DEFAULT_CFG)
    if cfg is not None:
        resolved.update(cfg)
    return resolved


def _build_base_transforms(
    cfg: Dict[str, Any], keys: Sequence[str] = (IMAGE_KEY,)
) -> list:
    """Return the deterministic (non-augmentation) transforms shared by
    both training and validation pipelines.

    The pipeline order is:
    1. **LoadImaged** – read NIfTI from disk.
    2. **EnsureChannelFirstd** – guarantee shape ``(C, D, H, W)``.
    3. **Spacingd** – resample to isotropic / target spacing.
    4. **ScaleIntensityRanged** – clip HU values to ``[intensity_min,
       intensity_max]`` and rescale to ``[0, 1]``.
    5. **ResizeWithPadOrCropd** – deterministic centre crop or zero-pad to
       ``spatial_size``.
    6. **NormalizeIntensityd** – channel-wise z-score-style normalization
       ``(x - mean) / std`` using the configured mean and std.

    Parameters
    ----------
    cfg : dict
        Resolved configuration dictionary.

    Returns
    -------
    list
        List of MONAI dictionary transforms.
    """
    spatial_size: Sequence[int] = cfg["spatial_size"]
    target_spacing: Sequence[float] = cfg["target_spacing"]
    intensity_min: float = cfg["intensity_min"]
    intensity_max: float = cfg["intensity_max"]
    normalize_mean: float = cfg["normalize_mean"]
    normalize_std: float = cfg["normalize_std"]

    return [
        # 1. Load NIfTI file from disk
        LoadImaged(keys=list(keys), image_only=True),
        # 2. Ensure channel-first layout (C, D, H, W)
        EnsureChannelFirstd(keys=list(keys)),
        # 3. Resample to uniform voxel spacing
        Spacingd(
            keys=list(keys),
            pixdim=target_spacing,
            mode="bilinear",
        ),
        # 4. Clip HU window and rescale to [0, 1]
        ScaleIntensityRanged(
            keys=list(keys),
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 5. Crop or pad to fixed spatial size
        ResizeWithPadOrCropd(keys=list(keys), spatial_size=spatial_size),
        # 6. Normalize: (x - mean) / std
        NormalizeIntensityd(
            keys=list(keys),
            subtrahend=normalize_mean,
            divisor=normalize_std,
        ),
    ]


def get_train_transforms(
    cfg: Dict[str, Any] | None = None,
    keys: Sequence[str] = (IMAGE_KEY,),
) -> Compose:
    """Build the **training** transform pipeline.

    Includes all base deterministic transforms followed by stochastic
    augmentations:

    * **RandFlipd** – random flip along each spatial axis (p=0.5 each).
    * **RandRotate90d** – random 90° rotation in a random spatial plane.
    * **RandGaussianNoised** – additive Gaussian noise (σ=0.05).
    * **RandShiftIntensityd** – random intensity offset in [-0.1, 0.1].
    * **RandAdjustContrastd** – random gamma contrast adjustment.
    * **RandGaussianSmoothd** – random Gaussian blurring.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.  See module docstring for expected keys.
        ``None`` uses built-in defaults.

    Returns
    -------
    monai.transforms.Compose
        Composed training transforms operating on dictionary data with
        key ``"image"``.

    Examples
    --------
    >>> from src.data.transforms import get_train_transforms
    >>> train_tfm = get_train_transforms({"spatial_size": [48, 96, 96]})
    >>> sample = {"image": "/data/patient_001/arterial.nii.gz"}
    >>> result = train_tfm(sample)
    >>> result["image"].shape  # torch.Size([1, 48, 96, 96])
    """
    cfg = _resolve_cfg(cfg)
    keys = list(keys)
    base = _build_base_transforms(cfg, keys)

    augmentations = [
        # Random flips along depth, height, width
        # A single dictionary transform randomizes once and applies the same
        # spatial operation to every registered sequence.
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        # Random 90° rotation
        RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(1, 2)),
        # Additive Gaussian noise
        RandGaussianNoised(keys=keys, prob=0.3, mean=0.0, std=0.05),
        # Random intensity shift
        RandShiftIntensityd(keys=keys, prob=0.3, offsets=0.1),
        # Random gamma contrast
        RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
        # Random Gaussian smoothing
        RandGaussianSmoothd(
            keys=keys,
            prob=0.2,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
    ]

    return Compose(base + augmentations)


def get_val_transforms(
    cfg: Dict[str, Any] | None = None,
    keys: Sequence[str] = (IMAGE_KEY,),
) -> Compose:
    """Build the **validation / test** transform pipeline.

    Identical to the training pipeline but **without** any stochastic
    augmentation, ensuring deterministic and reproducible inference.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.  See module docstring for expected keys.
        ``None`` uses built-in defaults.

    Returns
    -------
    monai.transforms.Compose
        Composed validation transforms operating on dictionary data with
        key ``"image"``.

    Examples
    --------
    >>> from src.data.transforms import get_val_transforms
    >>> val_tfm = get_val_transforms()
    >>> sample = {"image": "/data/patient_001/arterial.nii.gz"}
    >>> result = val_tfm(sample)
    >>> result["image"].shape  # torch.Size([1, 64, 128, 128])
    """
    cfg = _resolve_cfg(cfg)
    return Compose(_build_base_transforms(cfg, keys))
