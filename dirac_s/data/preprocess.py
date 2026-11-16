"""HU-aware intensity handling, body-region cropping, fixed-grid tensor formatting — spec Section 10
items 6-9. Float precision preserved throughout (no 8-bit quantization, unlike covtoken's `window()`).
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

HU_MIN, HU_MAX = -200.0, 400.0   # [CALIBRATE] — abdominal soft-tissue-adjacent range covering
                                  # kidney/liver/pancreas; wider than covtoken's WC=40/WW=400 tumor-only
                                  # window since DIRAC's pretraining needs broader anatomical context,
                                  # not just lesion-region contrast. Revisit once Phase 1 QC stats exist.
FOREGROUND_HU_THRESHOLD = -150.0   # air is ~-1000 HU; a generous threshold above air, below soft tissue


def clip_and_normalize_hu(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """Clips to [hu_min, hu_max], normalizes to [-1, 1] as float32 — no uint8 quantization."""
    clipped = np.clip(volume, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def foreground_bbox(volume: np.ndarray, threshold: float = FOREGROUND_HU_THRESHOLD,
                    margin_vox: int = 4) -> tuple[slice, slice, slice]:
    """Body-region bounding box from raw HU values (before normalization) — a simple foreground
    threshold, not a learned segmentation, so it stays unsupervised/label-free per spec Section 10 item 6."""
    mask = volume > threshold
    if not mask.any():
        return tuple(slice(0, s) for s in volume.shape)
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=0) + margin_vox + 1, volume.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def crop_to_bbox(volume: np.ndarray, bbox: tuple[slice, slice, slice]) -> np.ndarray:
    return volume[bbox]


def resize_to_fixed(volume: np.ndarray, target_shape: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Trilinear resize to a fixed grid (formalization Section 2's conv-stem input requirement:
    D=H=W divisible by 16)."""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = ndimage.zoom(volume, zoom_factors, order=1)
    out = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(t, r)) for t, r in zip(target_shape, resized.shape))
    out[slices] = resized[slices]
    return out


def preprocess_volume(raw_hu: np.ndarray, target_shape: tuple[int, int, int] = (128, 128, 128)) -> dict:
    """Full pipeline: foreground crop (raw HU) -> HU clip/normalize -> fixed-grid resize.
    Returns both the processed tensor and QC-relevant metadata (spec Section 10 item 2)."""
    bbox = foreground_bbox(raw_hu)
    cropped = crop_to_bbox(raw_hu, bbox)
    foreground_frac = float((cropped > FOREGROUND_HU_THRESHOLD).mean())
    normalized = clip_and_normalize_hu(cropped)
    resized = resize_to_fixed(normalized, target_shape)
    return {
        "volume": resized,
        "qc": {
            "original_shape": tuple(raw_hu.shape),
            "cropped_shape": tuple(cropped.shape),
            "foreground_fraction": foreground_frac,
            "hu_range_seen": (float(raw_hu.min()), float(raw_hu.max())),
        },
    }


def qc_pass(qc: dict, min_foreground_frac: float = 0.05) -> bool:
    """Gate 1's QC check (IMPLEMENTATION_SPEC.md Section 1): a case with near-zero foreground after
    cropping indicates a corrupt/empty/mis-windowed volume, not usable anatomy."""
    return qc["foreground_fraction"] >= min_foreground_frac
