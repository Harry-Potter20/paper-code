"""NIfTI load + orientation/spacing normalization — dirac_formalization.md Section 10 items 3-4 /
IMPLEMENTATION_SPEC.md Phase 1. Genuinely new code: covtoken's own CT ingestion (data/ct_bank.py,
jobs/materialize_kits_job.py) discards the 3D volume slice-by-slice into 8-bit RGB PNGs and never
reorients or resamples — verified directly against covtoken's actual code before writing this file.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import nibabel as nib
import nibabel.processing

from data.leak_guard import assert_label_free


@dataclass
class Volume:
    data: np.ndarray        # float32, HU values, canonical orientation, isotropic spacing
    spacing_mm: tuple[float, float, float]
    affine: np.ndarray


def load_canonical(path: str, target_spacing_mm: float = 1.5) -> Volume:
    """Load a NIfTI volume, reorient to closest canonical (RAS-ish), resample to isotropic spacing.
    target_spacing_mm=1.5 balances anatomical detail against the downstream fixed-grid tensor size
    (formalization Section 2's 128^3 target) — [CALIBRATE], to be revisited once real case statistics
    from Phase 1's QC pass are available."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    resampled = nibabel.processing.resample_to_output(
        img, voxel_sizes=[target_spacing_mm] * 3, order=1,   # order=1: trilinear, preserves HU better than nearest
    )
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    return Volume(data=data, spacing_mm=(target_spacing_mm,) * 3, affine=resampled.affine)


def load_canonical_mask(path: str, reference: Volume, target_spacing_mm: float = 1.5) -> np.ndarray:
    """Same reorientation/resampling as load_canonical, but nearest-neighbor (order=0) — masks are
    categorical labels, interpolating them would invent fractional label values. Eval-only per
    data/leak_guard.py; callers outside downstream-transfer probing must never call this."""
    assert_label_free("segmentation mask")
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    resampled = nibabel.processing.resample_to_output(
        img, voxel_sizes=[target_spacing_mm] * 3, order=0,
    )
    return np.asarray(resampled.dataobj, dtype=np.int16)
