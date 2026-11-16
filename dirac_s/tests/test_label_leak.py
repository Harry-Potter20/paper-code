"""CI gate: a segmentation label/mask must NEVER be accessible during self-supervised pretraining.
Ported from covtoken/tests/test_label_leak.py — same pattern, same contract, adapted to DIRAC's own
pretraining_guard() (dirac_formalization.md Section 10, IMPLEMENTATION_SPEC.md Phase 1 Gate 1).
"""
import pytest

from data.leak_guard import LabelLeakError, assert_label_free, pretraining_guard


def test_label_read_allowed_outside_pretraining():
    # EVAL paths (downstream-transfer probing, outside pretraining_guard) may read labels freely.
    assert_label_free()   # no raise


def test_label_read_blocked_during_pretraining():
    with pretraining_guard():
        with pytest.raises(LabelLeakError):
            assert_label_free()


def test_guard_is_reentrant_and_restores():
    with pretraining_guard():
        with pretraining_guard():
            with pytest.raises(LabelLeakError):
                assert_label_free()
    assert_label_free()   # outside again: no raise


def test_simulated_pretraining_step_with_leak_fails():
    """A pretraining step that illegally reads a label must blow up via the guard — the actual failure
    shape this test exists to catch, not just the guard's own raise/no-raise behavior in isolation."""

    def bad_pretrain_step(volume, label):
        with pretraining_guard():
            _ = assert_label_free("segmentation mask")   # ILLEGAL: this line represents a leak site
            return volume.mean() if hasattr(volume, "mean") else sum(volume) / len(volume)

    with pytest.raises(LabelLeakError):
        bad_pretrain_step([1.0, 2.0, 3.0], label=None)


def test_load_canonical_mask_calls_the_guard():
    """The real call site (data/volume_io.py's load_canonical_mask) must itself invoke the guard, not
    just have a guard module sitting unused elsewhere. Confirmed via monkeypatching, not by inspection."""
    import data.volume_io as volume_io

    called = {"count": 0}
    original = volume_io.assert_label_free

    def spy(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    volume_io.assert_label_free = spy
    try:
        with pretraining_guard():
            with pytest.raises(LabelLeakError):
                volume_io.load_canonical_mask("/nonexistent/path.nii.gz", reference=None)
        assert called["count"] == 1
    finally:
        volume_io.assert_label_free = original
