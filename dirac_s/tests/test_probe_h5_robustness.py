"""Phase 5 slice 6 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for H5 robustness -- same
discipline as tests/test_probe_h3_topology.py. Focuses on the pure-logic pieces (perturbation determinism/
range/shape, target alignment, probe scoring); the real forward-pass extraction is covered by a local dry
run against a tiny fixture, same as every other job in this project.
"""
from __future__ import annotations
import numpy as np
import torch

from jobs.probe_h5_robustness_job import (
    apply_perturbation, align, score, train_probe, LinearRegressionProbe, PERTURBATIONS,
)


def synthetic_volume():
    torch.manual_seed(0)
    return torch.clamp(torch.randn(1, 1, 128, 128, 128) * 0.3, -1.0, 1.0)


def test_all_perturbation_kinds_preserve_shape_and_range():
    x = synthetic_volume()
    device = torch.device("cpu")
    for name, (kind, param) in PERTURBATIONS.items():
        out = apply_perturbation(x, kind, param, device, seed=42)
        assert out.shape == x.shape, f"{name} changed shape"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"
        if kind in ("gaussian_noise", "intensity_shift"):
            assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5, f"{name} left the clamped range"


def test_perturbations_are_deterministic_given_the_same_seed():
    x = synthetic_volume()
    device = torch.device("cpu")
    for name, (kind, param) in PERTURBATIONS.items():
        out1 = apply_perturbation(x, kind, param, device, seed=123)
        out2 = apply_perturbation(x, kind, param, device, seed=123)
        assert torch.equal(out1, out2), f"{name} not deterministic given the same seed"


def test_gaussian_noise_actually_perturbs_the_input():
    x = synthetic_volume()
    device = torch.device("cpu")
    out = apply_perturbation(x, "gaussian_noise", 0.15, device, seed=1)
    assert not torch.equal(out, x)
    assert (out - x).abs().mean() > 0.01


def test_partial_dropout_zeros_roughly_the_requested_fraction():
    x = torch.ones(1, 1, 128, 128, 128)   # all-ones so zeroed voxels are trivially countable
    device = torch.device("cpu")
    out = apply_perturbation(x, "partial_dropout", 0.25, device, seed=1)
    zero_frac = (out == 0.0).float().mean().item()
    assert 0.15 < zero_frac < 0.35, f"expected ~25% zeroed, got {zero_frac:.2%}"


def test_align_filters_to_cases_with_a_target():
    rep = {"rep": np.random.randn(3, 8).astype(np.float32), "case_id": np.array(["a", "b", "c"])}
    targets = {"a": np.zeros(9, dtype=np.float32), "c": np.ones(9, dtype=np.float32)}
    out = align(rep, targets)
    assert list(out["case_id"]) == ["a", "c"]
    assert out["rep"].shape == (2, 8)
    assert out["target"].shape == (2, 9)


def test_score_handles_degenerate_dimension_and_recovers_synthetic_signal():
    rng = np.random.default_rng(0)
    n, dim_in, dim_out = 100, 32, 9
    x = rng.standard_normal((n, dim_in)).astype(np.float32)
    w = rng.standard_normal((dim_in, dim_out)).astype(np.float32)
    y = (x @ w + rng.standard_normal((n, dim_out)) * 0.05).astype(np.float32)
    y[:, -1] = 0.0   # degenerate dim

    device = torch.device("cpu")
    probe = train_probe(dim_in, dim_out, x, y, device)
    result = score(probe, x, y, device, seed=0)
    assert np.isfinite(result["macro_spearman"])
    assert result["macro_spearman"] > 0.8   # near-perfect recovery on the 8 non-degenerate dims
