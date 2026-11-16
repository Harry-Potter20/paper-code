"""Diagnostic, NOT a gate, NOT part of Section 5 -- does gudhi's persistence pairing on the decoder's own
reconstruction stay stable across consecutive real SGD steps, or does it churn?

Motivation (see CLAUDE.md, 2026-08-20): the real train_diff_topology_job.py run plateaus loss_topology_diff
around ~0.35-0.40 regardless of LAM_TD=1.0 vs 0.01 -- a 100x weight change barely moves the converged value,
which argues against a simple magnitude/calibration story. Formalization Section 2a's differentiability
argument ("the pairing structure is locally constant almost everywhere") is a POINTWISE guarantee -- true at
a single forward pass, but says nothing about whether a multi-step optimization trajectory that necessarily
crosses many of those local-constancy boundaries gets a *consistent* descent direction across steps. If the
active (birth_idx, death_idx) pairing set changes step-to-step, gradient earned under one pairing can become
irrelevant the moment gudhi re-pairs on the next forward pass -- exactly the "doesn't diverge, doesn't
improve either, insensitive to reweighting" signature the real run shows.

Mirrors tests/test_topology_diff.py::test_end_to_end_wiring_reaches_encoder's setup (same known-good small
config) but instead of one forward/backward, runs many real AdamW steps (same optimizer, same default LR as
train_diff_topology_job.py) on a FIXED input+target and inspects model.topology_diff._pairing_indices
directly, before/after each step, per homology dimension. CPU-only, no GPU, no HF Job, no bucket I/O -- a
local mechanism probe, cheap enough to just run.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from model.topology import compute_persistence_summary, TopologyHead
from model.topology_diff import _pairing_indices, differentiable_persistence_summary_batch
from model.dirac_model import DiracModel
from model.losses import Decoder3D, ZBroadcast, l_topology

MAX_DIM = 2
N_STEPS = 60
LR = 1e-4   # matches train_diff_topology_job.py's default


def _structured_volume(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((size, size, size)).astype(np.float32) * 0.3
    for _ in range(4):
        c = rng.integers(2, size - 2, size=3)
        r = rng.integers(2, max(3, size // 4))
        zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
        blob = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < r ** 2
        vol[blob] += 2.0
    return vol


def pairing_key(volume_2d_ready: torch.Tensor) -> list[set]:
    """Returns, per dim 0..MAX_DIM, the set of (birth_idx, death_idx) tuples currently active."""
    with torch.no_grad():
        neg_np = (-volume_2d_ready).detach().cpu().numpy().astype(np.float64)
    pairings = _pairing_indices(neg_np, MAX_DIM)
    return [set(zip(b.tolist(), d.tolist())) for b, d in pairings]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    torch.manual_seed(0)
    D, K, Z_DIM = 32, 4, 64
    VOLUME_SIZE, STEM_STRIDE = 64, 16
    GRID_SHAPE = (VOLUME_SIZE // STEM_STRIDE,) * 3
    TOPOLOGY_DOWNSAMPLE = 16

    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=MAX_DIM, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    params = list(model.parameters()) + list(decoder.parameters()) + list(z_broadcast.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    x_np = _structured_volume(VOLUME_SIZE, seed=1)
    x = torch.from_numpy(x_np)[None, None]
    ph_target = torch.from_numpy(
        compute_persistence_summary(
            ndimage.zoom(x_np, TOPOLOGY_DOWNSAMPLE / VOLUME_SIZE, order=1), max_dim=MAX_DIM
        )
    ).unsqueeze(0)
    ph_vector_dummy = torch.zeros(1, TopologyHead.summary_length(MAX_DIM))

    pairing_history = []   # list of (per-dim list of sets), one per step, captured BEFORE that step's update
    loss_history = []
    count_history = []     # per-dim pair COUNT per step -- cheaper instability signal than full set overlap

    for step in range(N_STEPS):
        out = model(x, ph_vector=ph_vector_dummy)
        decoder_in = z_broadcast(out["graph"].node_features, out["z"])
        recon = decoder(decoder_in, GRID_SHAPE)
        recon_small = F.interpolate(recon, size=(TOPOLOGY_DOWNSAMPLE,) * 3, mode="trilinear", align_corners=False)

        pairing_history.append(pairing_key(recon_small[0, 0]))
        count_history.append([len(s) for s in pairing_history[-1]])

        pred_summary = differentiable_persistence_summary_batch(recon_small, max_dim=MAX_DIM)
        loss = l_topology(pred_summary, ph_target)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[pairing-stability] {N_STEPS} real AdamW steps, lr={LR}, dims=0..{MAX_DIM}")
    print(f"[pairing-stability] loss_topology_diff: step0={loss_history[0]:.4f} "
          f"step{N_STEPS//2}={loss_history[N_STEPS//2]:.4f} step{N_STEPS-1}={loss_history[-1]:.4f}")

    for dim in range(MAX_DIM + 1):
        counts = [c[dim] for c in count_history]
        jaccards = [jaccard(pairing_history[t][dim], pairing_history[t + 1][dim]) for t in range(N_STEPS - 1)]
        exact_match = sum(1 for j in jaccards if j == 1.0)
        print(f"\n--- H{dim} ---")
        print(f"pair count over time: min={min(counts)} max={max(counts)} "
              f"first5={counts[:5]} last5={counts[-5:]}")
        print(f"consecutive-step Jaccard overlap: mean={np.mean(jaccards):.3f} "
              f"min={min(jaccards):.3f} steps_with_IDENTICAL_pairing={exact_match}/{N_STEPS-1}")

    print("\n[pairing-stability] interpretation: mean Jaccard close to 1.0 across all dims would mean the "
          "pairing is stable and this is NOT the mechanism behind the observed training-time plateau. Mean "
          "Jaccard well below 1.0, especially combined with pair-count churn, means the active pairing is "
          "genuinely unstable across steps -- gradient earned under one pairing routinely stops applying to "
          "the next step's pairing, which is consistent with (not provably the sole cause of, but consistent "
          "with) the observed insensitivity to LAM_TD reweighting.")


if __name__ == "__main__":
    main()
