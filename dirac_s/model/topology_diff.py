"""Differentiable persistence-summary loss on the decoder's own reconstruction —
dirac_diff_topology_formalization.md Section 2. NOT a Gate-6 file: this is new, ungated follow-on machinery,
never imported by pretrain_dirac_job.py or any signed-off ablation job.

Reuses gudhi.CubicalComplex exactly as model/topology.py::compute_persistence_summary does (same
superlevel-set convention, same [count, sum, max]-per-dimension, log1p-transformed layout, same summary
length), so `l_topology` (model/losses.py) can be reused verbatim against the exact same frozen `ph_vector`
target — the only thing this module changes is which side of the comparison is differentiable.

Mechanism (formalization Section 2a, live-verified against gudhi 3.13.0 before writing this, not assumed):
`gudhi.CubicalComplex(top_dimensional_cells=...)` + `cofaces_of_persistence_pairs()` reports, per homology
dimension, `(birth_idx, death_idx)` pairs that index directly into the *flattened original array* — for
H0, H1, and H2 alike. So no reimplementation of persistence and no custom autograd.Function is needed: the
gudhi call runs once per forward pass under no_grad (to get the pairing structure only), then plain PyTorch
fancy-indexing on the differentiable tensor recovers the same birth/death values with a real gradient path.
"""
from __future__ import annotations
import numpy as np
import torch
import gudhi

from model.topology import TopologyHead


def _pairing_indices(volume_np: np.ndarray, max_dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """volume_np: 3D float64 array, already negated to match compute_persistence_summary's superlevel-set
    convention (caller's responsibility, mirroring that function exactly). Returns, per dimension 0..max_dim,
    a (birth_idx, death_idx) pair of int arrays into volume_np.flatten() — gudhi-computed, not differentiated.
    """
    cc = gudhi.CubicalComplex(top_dimensional_cells=volume_np.flatten(), dimensions=list(volume_np.shape))
    cc.compute_persistence()
    coface_pairs = cc.cofaces_of_persistence_pairs()[0]   # index 0 = finite pairs (already excludes the essential/infinite class at index 1 — matches compute_persistence_summary's np.isfinite filter)
    out = []
    for dim in range(max_dim + 1):
        pairs_dim = coface_pairs[dim] if dim < len(coface_pairs) else np.zeros((0, 2), dtype=np.int64)
        out.append((pairs_dim[:, 0].astype(np.int64), pairs_dim[:, 1].astype(np.int64)))
    return out


def differentiable_persistence_summary(volume: torch.Tensor, max_dim: int = 2) -> torch.Tensor:
    """volume: (D, H, W) real-valued torch tensor (requires_grad-capable — typically a downsampled decoder
    reconstruction, formalization Section 2b). Returns (max_dim+1)*3, matching
    TopologyHead.summary_length(max_dim) exactly, gradient-carrying w.r.t. `volume`.

    Single-volume only (gudhi has no batch API) — callers loop over the batch dimension, same pattern
    _compute_one_ph (data/local_cache.py) already uses per-case."""
    assert volume.dim() == 3, f"expected (D,H,W), got {tuple(volume.shape)}"
    with torch.no_grad():
        neg_np = (-volume).detach().cpu().numpy().astype(np.float64)
        pairings = _pairing_indices(neg_np, max_dim)

    flat = (-volume).flatten()   # differentiable; same superlevel-set negation as the frozen target, applied here too
    stats = []
    for birth_idx, death_idx in pairings:
        if len(birth_idx) == 0:
            stats.extend([flat.new_zeros(()), flat.new_zeros(()), flat.new_zeros(())])
            continue
        b_idx_t = torch.as_tensor(birth_idx, device=volume.device, dtype=torch.long)
        d_idx_t = torch.as_tensor(death_idx, device=volume.device, dtype=torch.long)
        lifetimes = flat[d_idx_t] - flat[b_idx_t]                       # differentiable gather + subtract
        lifetimes = torch.clamp(lifetimes, min=0.0)                     # numerical guard only — see module docstring / formalization Sec 2c
        count = flat.new_tensor(float(len(birth_idx)))                  # non-differentiable by construction (Sec 2c) — fixed pairing cardinality
        stats.extend([count, lifetimes.sum(), lifetimes.max()])
    summary = torch.stack(stats)
    return torch.log1p(summary)


def differentiable_persistence_summary_batch(volumes: torch.Tensor, max_dim: int = 2) -> torch.Tensor:
    """volumes: (B, D, H, W) or (B, 1, D, H, W). Returns (B, (max_dim+1)*3)."""
    if volumes.dim() == 5:
        volumes = volumes.squeeze(1)
    assert volumes.dim() == 4, f"expected (B,D,H,W) or (B,1,D,H,W), got {tuple(volumes.shape)}"
    return torch.stack([differentiable_persistence_summary(v, max_dim=max_dim) for v in volumes])
