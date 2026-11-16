"""Phase 5 slice 3 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for the H3-A/H3-B
geometry/relations probes -- same discipline as test_probe_h3_topology.py.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from jobs.probe_h3ab_geometry_relations_job import (
    geometry_and_relation_targets, attach_targets, evaluate, GEOMETRY_DIM_NAMES, RELATION_DIM_NAMES,
)


def _fake_mask(tmp_path: Path, name: str, shape=(20, 20, 20), organ_box=None, tumor_box=None) -> str:
    mask = np.zeros(shape, dtype=np.uint8)
    if organ_box is not None:
        (z0, z1), (y0, y1), (x0, x1) = organ_box
        mask[z0:z1, y0:y1, x0:x1] = 1
    if tumor_box is not None:
        (z0, z1), (y0, y1), (x0, x1) = tumor_box
        mask[z0:z1, y0:y1, x0:x1] = 2   # tumor drawn after organ, so it can sit "inside" it
    path = tmp_path / name
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(path))
    return str(path)


def test_geometry_target_is_none_when_tumor_or_organ_missing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # organ only, no tumor
        path = _fake_mask(tmp_path, "no_tumor.nii.gz", organ_box=((0, 5), (0, 5), (0, 5)), tumor_box=None)
        geom, rel = geometry_and_relation_targets({"local_seg_path": path})
        assert geom is None and rel is None


def test_geometry_and_relation_targets_are_fractional_and_scale_invariant():
    """A tumor at a known location/size in a KNOWN shape should give predictable fractional centroid/extent
    -- and re-running the same box at 2x the mask resolution (same relative geometry) should give the SAME
    fractional targets, confirming the scale-invariance the docstring claims (the whole point of using
    fractions instead of absolute voxel/mm units, given the H1 audit's confirmed per-case-zoom confound)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # small mask: organ fills [0:20], tumor is [10:20] along z, centered in y/x -- known centroid/extent
        path_small = _fake_mask(tmp_path, "small.nii.gz", shape=(20, 20, 20),
                                organ_box=((0, 20), (0, 20), (0, 20)), tumor_box=((10, 20), (5, 15), (5, 15)))
        path_big = _fake_mask(tmp_path, "big.nii.gz", shape=(40, 40, 40),
                              organ_box=((0, 40), (0, 40), (0, 40)), tumor_box=((20, 40), (10, 30), (10, 30)))

        geom_small, rel_small = geometry_and_relation_targets({"local_seg_path": path_small})
        geom_big, rel_big = geometry_and_relation_targets({"local_seg_path": path_big})

        # Not bit-exact: discrete voxel-index centroids have a half-voxel offset that doesn't cancel
        # exactly when box size doubles (e.g. mean(10..19)/20=0.725 vs mean(20..39)/40=0.7375). atol=0.02
        # comfortably covers that expected discretization effect while still confirming approximate
        # scale invariance -- the real point of this test.
        np.testing.assert_allclose(geom_small, geom_big, atol=0.02)
        np.testing.assert_allclose(rel_small, rel_big, atol=0.02)
        # sanity: tumor spans z in [10,20) of a 20-long axis -> extent_z fraction = 10/20 = 0.5
        assert abs(geom_small[GEOMETRY_DIM_NAMES.index("extent_z")] - 0.5) < 1e-6


def test_attach_targets_filters_to_cases_with_a_target():
    rep = {"train": {"rep": np.random.randn(3, 8).astype(np.float32), "case_id": np.array(["a", "b", "c"])}}
    targets = {"a": np.zeros(6, dtype=np.float32), "c": np.ones(6, dtype=np.float32)}
    out = attach_targets(rep, targets)
    assert list(out["train"]["case_id"]) == ["a", "c"]
    assert out["train"]["target"].shape == (2, 6)


def test_evaluate_recovers_synthetic_linear_geometry_signal():
    rng = np.random.default_rng(0)
    n_train, n_val, dim_in = 80, 30, 32
    w = rng.standard_normal((dim_in, 6)).astype(np.float32)
    x_train = rng.standard_normal((n_train, dim_in)).astype(np.float32)
    x_val = rng.standard_normal((n_val, dim_in)).astype(np.float32)
    data = {
        "train": {"rep": x_train, "target": (x_train @ w + rng.standard_normal((n_train, 6)) * 0.05).astype(np.float32)},
        "val": {"rep": x_val, "target": (x_val @ w + rng.standard_normal((n_val, 6)) * 0.05).astype(np.float32)},
    }
    result = evaluate("test", GEOMETRY_DIM_NAMES, data, torch.device("cpu"))
    assert result["macro_spearman"] > 0.8
