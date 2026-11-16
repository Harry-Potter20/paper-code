"""Gate 0 (IMPLEMENTATION_SPEC.md Section 1, row 0 / row 4): N_DIRAC-S, N_CNN, N_ViT computed exactly and
within the locked 5% tolerance (formalization Section 9 / Section 12 falsification gate 6 — "the
contribution is the structure, not the parameter count," checked).
"""
from tests._common import build_model, VOLUME_SIZE
from model.param_count import exact_param_count, parameter_ledger, assert_no_complex_dtype_params
from baselines.cnn3d import CNN3DBaseline
from baselines.vit3d import ViT3DBaseline

TOLERANCE_PCT = 5.0   # [FIXED], per IMPLEMENTATION_SPEC.md Gate 0 row


def test_no_complex_dtype_params_anywhere():
    """Guards the counting convention itself (formalization Section 9): complex quantities must stay
    (re, im) real-Linear pairs, never a torch.complex-dtype parameter, or numel() silently undercounts."""
    assert_no_complex_dtype_params(build_model())
    assert_no_complex_dtype_params(CNN3DBaseline())
    assert_no_complex_dtype_params(ViT3DBaseline(volume_size=VOLUME_SIZE))


def test_ledger_buckets_sum_to_total():
    ledger = parameter_ledger(build_model())
    bucket_sum = sum(v for k, v in ledger.items() if k not in ("total",))
    assert bucket_sum == ledger["total"]


def test_dirac_cnn_vit_within_matching_tolerance():
    n_dirac = exact_param_count(build_model())
    n_cnn = exact_param_count(CNN3DBaseline())
    n_vit = exact_param_count(ViT3DBaseline(volume_size=VOLUME_SIZE))

    delta_cnn_pct = abs(n_dirac - n_cnn) / n_dirac * 100
    delta_vit_pct = abs(n_dirac - n_vit) / n_dirac * 100

    assert delta_cnn_pct <= TOLERANCE_PCT, f"CNN off by {delta_cnn_pct:.2f}% (N_DIRAC={n_dirac}, N_CNN={n_cnn})"
    assert delta_vit_pct <= TOLERANCE_PCT, f"ViT off by {delta_vit_pct:.2f}% (N_DIRAC={n_dirac}, N_ViT={n_vit})"
