"""Phase 5 slice 1 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for H1 probing -- same
discipline as tests/test_pretrain_losses.py and tests/test_baseline_pretrain.py: catch wiring bugs (e.g. a
dict keyed one way and looked up another) before any GPU spend, not after. Several tests here exist
specifically because a pre-lock audit (prompted by the first real run's near-chance result) found two real
label bugs -- these tests are regression tests for those exact bugs, not just general coverage.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from eval.stats import auroc, delong_auc_ci, delong_auc_diff_test, paired_bootstrap_diff
from jobs.probe_linear_job import (
    LinearProbe, train_probe as train_linear, score as score_linear,
    tumor_volume_mm3, compute_labels_and_volumes,
    train_regression_probe, predict_regression, r_squared, pearson_r,
)
from jobs.probe_nonlinear_job import MLPProbe, train_probe as train_mlp, score as score_mlp


def test_auroc_hand_computable_cases():
    # perfect separation
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    # perfectly wrong ordering
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    # chance
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) - 0.5) < 1e-9


def test_delong_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 40 + [1] * 40)
    scores = labels * 0.6 + rng.random(80) * 0.5
    auc, (lo, hi) = delong_auc_ci(scores, labels)
    assert lo <= auc <= hi


def test_delong_diff_test_identical_scores_gives_zero_diff():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 30 + [1] * 30)
    scores = labels * 0.5 + rng.random(60) * 0.5
    result = delong_auc_diff_test(scores, scores, labels)
    assert abs(result["diff"]) < 1e-9
    assert result["ci95"][0] <= 0.0 <= result["ci95"][1]


def test_paired_bootstrap_diff_excludes_zero_for_clearly_different_arrays():
    a = np.full(50, 0.9)
    b = np.full(50, 0.1)
    result = paired_bootstrap_diff(a, b, n=500)
    assert result["excludes_0"]
    assert result["diff"] > 0.5


def test_delong_ci_degenerate_class_returns_nan_not_crash():
    """Regression test: the first real H1 run hit a ZeroDivisionError here because a tumor-presence label
    turned out ~100% positive on val, so the negative class was empty. Confirms the fix (a zero-guard,
    mirroring auroc()'s own pos==0/neg==0 check) reports NaN instead of crashing."""
    all_positive = np.ones(20, dtype=int)
    scores = np.random.default_rng(0).random(20)
    auc, ci = delong_auc_ci(scores, all_positive)
    assert np.isnan(auc)
    assert np.isnan(ci[0]) and np.isnan(ci[1])

    result = delong_auc_diff_test(scores, scores, all_positive)
    assert np.isnan(result["diff"])


def _fake_segmentation(tmp_path: Path, name: str, tumor_voxels: int, zooms=(1.0, 1.0, 1.0)) -> str:
    """Builds a tiny synthetic NIfTI mask with `tumor_voxels` voxels set to label 2, at a chosen physical
    voxel size (zooms, in mm) -- lets tests control physical volume independently of voxel count."""
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=np.uint8)
    mask.flat[:tumor_voxels] = 2
    affine = np.diag([*zooms, 1.0])
    img = nib.Nifti1Image(mask, affine)
    path = tmp_path / name
    nib.save(img, str(path))
    return str(path)


def test_tumor_volume_mm3_uses_physical_voxel_size_not_raw_count():
    """Regression test: the first real H1 run used raw voxel count as the label, uncontrolled for each
    scan's native voxel spacing (segmentation.nii.gz is never resampled, unlike volume.npy). Two masks
    with the SAME voxel count but DIFFERENT physical voxel sizes must give different mm^3 volumes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fine = _fake_segmentation(tmp_path, "fine.nii.gz", tumor_voxels=100, zooms=(0.5, 0.5, 0.5))
        coarse = _fake_segmentation(tmp_path, "coarse.nii.gz", tumor_voxels=100, zooms=(2.0, 2.0, 2.0))

        v_fine = tumor_volume_mm3({"local_seg_path": fine})
        v_coarse = tumor_volume_mm3({"local_seg_path": coarse})

        assert v_fine == 100 * 0.5 ** 3
        assert v_coarse == 100 * 2.0 ** 3
        assert v_coarse > v_fine   # same voxel count, but coarse voxels are physically much bigger


def test_median_threshold_is_train_only_not_leaked_from_val():
    """Regression test: the first real H1 run computed the per-dataset median over pooled train+val
    volumes, leaking validation tumor volumes into the very threshold used to label validation. Construct
    a case where train and val have deliberately different distributions -- if the median were computed on
    the pooled set (the bug), the val split would end up almost entirely in one class; computed correctly
    (train-only), val case labels are just "is this val case's volume above the TRAIN median," independent
    of what other val cases look like."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = []
        # train: volumes 10..90 (median 50)
        for i, vox in enumerate(range(10, 100, 10)):
            path = _fake_segmentation(tmp_path, f"train_{i}.nii.gz", tumor_voxels=vox)
            rows.append({"case_id": f"train_{i}", "dataset": "kits", "split": "train", "local_seg_path": path})
        # val: every case has a huge volume (1000) -- if the pooled median were used, these would all
        # trivially land "above median"; that's fine either way here, but a single deliberately-huge outlier
        # val case must NOT shift where the train cases land.
        val_path = _fake_segmentation(tmp_path, "val_0.nii.gz", tumor_voxels=1000)
        rows.append({"case_id": "val_0", "dataset": "kits", "split": "val", "local_seg_path": val_path})

        labels, volumes = compute_labels_and_volumes(rows)

        # train median is 50 (the 5th of 9 sorted values: 10,20,...,90) -- computed from train rows ONLY
        train_volumes = sorted(v for cid, v in volumes.items() if cid.startswith("train_"))
        assert train_volumes == [float(v) for v in range(10, 100, 10)]
        expected_median = 50.0
        for i, vox in enumerate(range(10, 100, 10)):
            assert labels[f"train_{i}"] == int(vox > expected_median), f"train_{i} (vox={vox}) mislabeled"
        # the val case's own extreme volume must never have been able to shift the train-derived threshold
        assert labels["val_0"] == 1   # 1000 > 50, correctly above the train-only median


def test_regression_probe_recovers_synthetic_linear_signal():
    rng = np.random.default_rng(0)
    n, z_dim = 200, 256
    z = rng.standard_normal((n, z_dim)).astype(np.float32)
    y = (z[:, 0] * 2.0 + rng.standard_normal(n) * 0.1).astype(np.float32)   # strong linear signal in dim 0
    device = torch.device("cpu")
    probe = train_regression_probe(LinearProbe(), z, y, device)
    pred = predict_regression(probe, z, device)
    assert pearson_r(pred, y) > 0.9
    assert r_squared(y, pred) > 0.8


def _separable_synthetic_features(n: int = 200, z_dim: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    z = rng.standard_normal((n, z_dim)).astype(np.float32)
    z[:, 0] += labels * 3.0   # one dimension carries a strong, linearly-separable signal
    return z, labels.astype(np.int64)


def test_linear_probe_learns_separable_synthetic_signal():
    z, labels = _separable_synthetic_features()
    device = torch.device("cpu")
    probe = train_linear(LinearProbe(), z, labels, device)
    s = score_linear(probe, z, device)
    assert auroc(s, labels) > 0.9


def test_mlp_probe_learns_separable_synthetic_signal():
    z, labels = _separable_synthetic_features()
    device = torch.device("cpu")
    probe = train_mlp(MLPProbe(), z, labels, device)
    s = score_mlp(probe, z, device)
    assert auroc(s, labels) > 0.9
