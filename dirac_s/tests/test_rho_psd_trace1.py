"""Gate 0 (IMPLEMENTATION_SPEC.md Section 1, row 0): forward pass runs end-to-end with correct complex
dtype/shape; every rho_i is numerically PSD and trace-1; T's PH summary is finite. All three share one
forward pass's setup, hence one file (formalization Section 12, falsification gates 1 and 5).
"""
import torch

from tests._common import build_model, synthetic_batch, BATCH, GRID_SHAPE, D, K


def test_forward_pass_shapes_and_dtype():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]

    assert out["psi"].re.shape == (BATCH, n, D)
    assert out["psi"].im.shape == (BATCH, n, D)
    assert out["rho"].weights.shape == (BATCH, n, K)
    assert out["rho"].basis.re.shape == (BATCH, n, K, D)
    assert out["z"].shape[0] == BATCH


def test_rho_is_psd_and_trace1():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    check = out["rho"].check_psd_trace1()
    assert check["psd_ok"], f"min eigenvalue {check['min_eig']} below tolerance"
    assert check["trace_ok"], f"max trace error {check['max_trace_error']} above tolerance"


def test_rho_entropy_matches_dense_eigendecomposition():
    """Cross-validates entropy()'s efficient K x K path against a brute-force eigendecomposition of the
    materialized dense rho — independent verification of the two computational paths, per
    IMPLEMENTATION_SPEC Section 0 rule 7."""
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    rho = out["rho"]

    fast_entropy = rho.entropy()
    dense = rho.to_dense()
    eigvals = torch.linalg.eigvalsh(dense).clamp_min(0.0)
    brute_entropy = -(eigvals * torch.log(eigvals + 1e-12)).sum(-1)

    assert torch.allclose(fast_entropy, brute_entropy, atol=1e-3), (
        f"max abs diff {(fast_entropy - brute_entropy).abs().max().item()}"
    )


def test_pure_state_entropy_is_zero():
    """Construction A (formalization Section 3) — confirms the rejected pure-state construction's
    entropy really is identically zero, the exact reason it was rejected as the default."""
    from model.density_state import pure_state
    from model.complex_encoder import ComplexState
    psi = ComplexState(re=torch.randn(2, 10, D), im=torch.randn(2, 10, D))
    rho = pure_state(psi)
    assert torch.allclose(rho.entropy(), torch.zeros(2, 10), atol=1e-5)


def test_topology_summary_is_finite():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    assert torch.isfinite(out["topology"]).all()
    assert torch.isfinite(ph_vectors).all()
