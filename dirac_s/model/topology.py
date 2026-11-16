"""T(X) — dirac_formalization.md Section 5. Computed on the raw preprocessed HU volume (downsampled),
never on rho — computing persistent homology from an untrained rho early in training is a chicken-and-egg
problem (topology of noise). Same gudhi.CubicalComplex + ph_feature() recipe already proven in this vault
(topology/jobs/topology_gate0_job.py), generalized from 2D to 3D and extended to H0/H1/H2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import gudhi


@dataclass
class TopologySummary:
    ph_vector: torch.Tensor   # (B, m), m = n_dims * n_stats, fixed length


def compute_persistence_summary(volume: np.ndarray, max_dim: int = 2) -> np.ndarray:
    """volume: 3D scalar array (Dz, Hy, Wx), already downsampled (~32^3-64^3). Superlevel-set convention
    (negate before feeding gudhi's sublevel-set filtration), matching this vault's existing
    superlevel_persistence usage elsewhere. Returns a flat (n_dims * 3,) vector: [count, sum, max] of
    finite persistence lifetimes per homology dimension 0..max_dim, each log1p-transformed.

    log1p, not raw: count/sum can run into the thousands-tens-of-thousands even on Gate 0's own synthetic
    Gaussian-noise fixture (verified directly: ~5600 on a 32^3 standard-normal volume) — never caught by
    Gate 0/2, since "is it finite" was the only bar either gate checked, and neither gate ever computed a
    loss/gradient against this vector. Phase 3 is the first phase that does (L_topology's regression target,
    and TopologyHead's embedding input feeding z): an unnormalized target that large would dominate every
    other O(1) loss term and destabilize training, caught by pretrain_dirac_job.py's smoke run before any
    real GPU spend. All three raw statistics are non-negative by construction (count/sum/max of persistence
    lifetimes, which are themselves non-negative for a well-formed filtration), so log1p is safe everywhere
    without a sign-handling special case."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=(-volume).flatten().astype(np.float64),
                              dimensions=list(volume.shape))
    cc.compute_persistence()
    stats = []
    for dim in range(max_dim + 1):
        pairs = cc.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            stats.extend([0.0, 0.0, 0.0])
            continue
        lifetimes = pairs[:, 1] - pairs[:, 0]
        finite = lifetimes[np.isfinite(lifetimes)]
        if len(finite) == 0:
            stats.extend([0.0, 0.0, 0.0])
        else:
            stats.extend([float(len(finite)), float(finite.sum()), float(finite.max())])
    return np.log1p(np.array(stats, dtype=np.float32))


class TopologyHead(nn.Module):
    """Embeds a precomputed PH summary vector (from compute_persistence_summary, run outside autograd —
    gudhi is not differentiated through, per formalization Section 7's L_topology design) into the
    representation. `m` must match compute_persistence_summary's output length: (max_dim+1)*3."""

    def __init__(self, m: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(m, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, ph_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(ph_vector)

    @staticmethod
    def summary_length(max_dim: int = 2) -> int:
        return (max_dim + 1) * 3
