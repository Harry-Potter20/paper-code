"""Phase 5 slice 2 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for the H3-C topology probe --
same discipline as tests/test_probing.py. Focuses on the pure-logic pieces (target attachment, probe
training/eval, the degenerate-dimension guard); extract_dirac/extract_baseline's file I/O is covered by a
local dry run against a tiny fixture, same as every other job in this project.
"""
from __future__ import annotations
import numpy as np
import torch

from jobs.probe_h3_topology_job import (
    LinearRegressionProbe, train_probe, predict, r_squared, attach_targets, evaluate_condition, PH_DIM_NAMES,
)


def test_ph_dim_names_matches_compute_persistence_summary_layout():
    # 3 homology dims (H0,H1,H2) x 3 stats (count,sum,max) = 9, matching model/topology.py's
    # compute_persistence_summary(max_dim=2) output layout exactly -- no new target invented.
    assert PH_DIM_NAMES == ["H0_count", "H0_sum", "H0_max", "H1_count", "H1_sum", "H1_max",
                            "H2_count", "H2_sum", "H2_max"]


def test_attach_targets_filters_to_cases_with_a_target():
    rep = {"train": {"rep": np.random.randn(3, 8).astype(np.float32), "case_id": np.array(["a", "b", "c"])}}
    targets = {"a": np.zeros(9, dtype=np.float32), "c": np.ones(9, dtype=np.float32)}   # "b" has no target
    out = attach_targets(rep, targets)
    assert list(out["train"]["case_id"]) == ["a", "c"]
    assert out["train"]["rep"].shape == (2, 8)
    assert out["train"]["target"].shape == (2, 9)


def test_regression_probe_recovers_synthetic_multioutput_linear_signal():
    rng = np.random.default_rng(0)
    n, dim_in, dim_out = 200, 64, 9
    x = rng.standard_normal((n, dim_in)).astype(np.float32)
    w = rng.standard_normal((dim_in, dim_out)).astype(np.float32)
    y = (x @ w + rng.standard_normal((n, dim_out)) * 0.1).astype(np.float32)
    device = torch.device("cpu")
    probe = train_probe(dim_in, dim_out, x, y, device)
    pred = predict(probe, x, device)
    for j in range(dim_out):
        assert r_squared(y[:, j], pred[:, j]) > 0.8, f"dim {j} R^2 too low"


def test_evaluate_condition_handles_degenerate_target_dimension_gracefully():
    """A target dimension with zero variance across val (e.g. no H2 features in any sampled case) must be
    skipped with a note, not crash or emit a garbage Spearman correlation."""
    rng = np.random.default_rng(0)
    n_train, n_val, dim_in = 60, 30, 16
    x_train = rng.standard_normal((n_train, dim_in)).astype(np.float32)
    x_val = rng.standard_normal((n_val, dim_in)).astype(np.float32)
    y_train = rng.standard_normal((n_train, 9)).astype(np.float32)
    y_val = rng.standard_normal((n_val, 9)).astype(np.float32)
    y_train[:, -1] = 0.0   # H2_max degenerate: always zero
    y_val[:, -1] = 0.0

    data = {
        "train": {"rep": x_train, "target": y_train},
        "val": {"rep": x_val, "target": y_val},
    }
    device = torch.device("cpu")
    result = evaluate_condition("synthetic", data, device)

    assert result["per_dim"]["H2_max"]["spearman_rho"] is None
    assert "degenerate" in result["per_dim"]["H2_max"]["note"]
    # the other 8 dims should still have been evaluated normally
    assert result["per_dim"]["H0_count"]["spearman_rho"] is not None
    assert np.isfinite(result["macro_spearman"])
