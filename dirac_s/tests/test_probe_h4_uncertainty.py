"""Phase 5 slice 5 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for H4's uncertainty /
magnitude-conflation check -- same discipline as the other probe test files.
"""
from __future__ import annotations
import numpy as np
import torch

from model.complex_encoder import ComplexState
from eval.stats import partial_spearman
from jobs.probe_h4_uncertainty_job import categorical_entropy, normalize_psi, per_token_recon_error


def test_partial_spearman_removes_a_pure_confound():
    """If y is entirely explained by z (y = 2z + noise) and x is unrelated to y once z is removed, the
    RAW correlation of x with y can be large (both driven by the same z), but the PARTIAL correlation
    (controlling for z) should collapse toward zero -- this is the exact "magnitude conflation" pattern
    Section 8 checks for (S(rho) looking informative only because it, like the error, both track ||psi||)."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal(n)
    y = 2 * z + rng.standard_normal(n) * 0.1
    x = 3 * z + rng.standard_normal(n) * 0.1   # x is confounded with y purely through z, no direct link

    from scipy import stats as scipy_stats
    raw = float(scipy_stats.spearmanr(x, y).statistic)
    partial = partial_spearman(x, y, controlling_for=z)

    assert raw > 0.9   # strong raw correlation (both driven by the same confound)
    assert abs(partial) < 0.2   # collapses once the confound is controlled for


def test_partial_spearman_preserves_a_genuine_direct_relationship():
    """If x truly predicts y beyond what z explains, the partial correlation should stay strong, not
    collapse -- the "positive evidence against magnitude-conflation" case (Section 8 point 4)."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = 2 * x + 0.5 * z + rng.standard_normal(n) * 0.1   # genuine x->y link, not just through z

    partial = partial_spearman(x, y, controlling_for=z)
    assert partial > 0.8


def test_categorical_entropy_matches_known_cases():
    uniform = torch.full((1, 1, 4), 0.25)
    onehot = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    h_uniform = categorical_entropy(uniform)
    h_onehot = categorical_entropy(onehot)
    assert abs(h_uniform.item() - np.log(4)) < 1e-5
    assert abs(h_onehot.item() - 0.0) < 1e-5


def test_normalize_psi_gives_unit_norm():
    psi = ComplexState(re=torch.randn(2, 5, 8) * 3.0, im=torch.randn(2, 5, 8) * 3.0)
    normalized = normalize_psi(psi)
    norms = normalized.norm()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_per_token_recon_error_shape_and_zero_when_identical():
    x = torch.randn(2, 1, 32, 32, 32)
    err_zero = per_token_recon_error(x, x, stem_stride=16)
    assert err_zero.shape == (2, 8)   # (32/16)^3 = 8 tokens
    assert torch.allclose(err_zero, torch.zeros_like(err_zero))

    recon = x + 1.0
    err_one = per_token_recon_error(recon, x, stem_stride=16)
    assert torch.allclose(err_one, torch.ones_like(err_one), atol=1e-5)
