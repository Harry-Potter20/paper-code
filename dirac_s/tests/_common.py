"""Shared synthetic-input setup for Gate 0 (IMPLEMENTATION_SPEC.md Section 1) — not itself a test file."""
from __future__ import annotations
import numpy as np
import torch

from model.dirac_model import DiracModel
from model.topology import compute_persistence_summary, TopologyHead

D = 64          # embedding width, [CALIBRATE] default per formalization Section 1
K = 8           # mixture components, [CALIBRATE]
VOLUME_SIZE = 64   # synthetic input size (real target is 128; 64 keeps Gate 0 fast — divisible by 16)
GRID_SHAPE = (VOLUME_SIZE // 16,) * 3   # 4^3 = 64 tokens at volume_size=64
BATCH = 2


def build_model() -> DiracModel:
    return DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=2)


def synthetic_batch(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(BATCH, 1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
    rng = np.random.default_rng(seed)
    volumes = rng.standard_normal((BATCH, 32, 32, 32)).astype(np.float32)   # downsampled, per formalization Section 5
    m = TopologyHead.summary_length(max_dim=2)
    ph_vectors = np.stack([compute_persistence_summary(v, max_dim=2) for v in volumes])
    assert ph_vectors.shape == (BATCH, m)
    return x, torch.from_numpy(ph_vectors)
