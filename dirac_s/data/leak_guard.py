"""Runtime guard enforcing DIRAC's own label-free invariant: segmentation labels (KiTS/MSD) are
EVAL-ONLY, used only for downstream-transfer probing (spec Section 15 Task A), never touching
self-supervised pretraining (Objectives A-D are all label-free by construction — reconstruction, masking,
structural consistency, topology). Ported near-verbatim from covtoken/data/leak_guard.py — same pattern,
same contract, renamed for DIRAC's context (pretraining, not subspace construction).
"""
from __future__ import annotations
import threading
from contextlib import contextmanager

_state = threading.local()


class LabelLeakError(RuntimeError):
    """Raised when a segmentation label/mask is accessed during self-supervised pretraining."""


def _in_pretraining() -> bool:
    return getattr(_state, "depth", 0) > 0


@contextmanager
def pretraining_guard():
    """Mark a region as label-free self-supervised pretraining. Label/mask loading inside this region
    is a spec violation and raises."""
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def assert_label_free(what: str = "segmentation label/mask") -> None:
    """Call this at every label/mask read site. Raises inside pretraining_guard()."""
    if _in_pretraining():
        raise LabelLeakError(
            f"Spec violation: attempted to access {what} during label-free self-supervised "
            f"pretraining. Segmentation labels are EVAL-ONLY (dirac_formalization.md Section 10, "
            f"CLAUDE.md). Move this access outside pretraining_guard() — it belongs in downstream "
            f"transfer probing (spec Section 15), not pretraining (spec Section 11)."
        )
