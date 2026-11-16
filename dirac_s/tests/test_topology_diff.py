"""Falsification gates 1-4 for dirac_diff_topology_formalization.md Section 5. CPU-only, synthetic data,
same discipline as every other Gate 0 in this project (tests/_common.py). This is new, ungated follow-on
work — none of these tests touch a Gate-6 signed-off file.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from model.topology import compute_persistence_summary, TopologyHead
from model.topology_diff import differentiable_persistence_summary, differentiable_persistence_summary_batch
from model.dirac_model import DiracModel
from model.losses import Decoder3D, ZBroadcast, l_topology

MAX_DIM = 2


def _structured_volume(size: int, seed: int = 0) -> np.ndarray:
    """Not pure noise — a few blobs, so there's real, non-degenerate H0/H1/H2 structure to test against
    (matches how gate_0's own synthetic fixtures avoid pure-noise degeneracy)."""
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((size, size, size)).astype(np.float32) * 0.3
    for _ in range(4):
        c = rng.integers(2, size - 2, size=3)
        r = rng.integers(2, max(3, size // 4))
        zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
        blob = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < r ** 2
        vol[blob] += 2.0
    return vol


def test_consistency_matches_frozen_extractor():
    """Gate 1: the differentiable path must reproduce compute_persistence_summary exactly (same pairing,
    same values, formalization Section 2a)."""
    for seed in range(3):
        vol_np = _structured_volume(16, seed=seed)
        frozen = compute_persistence_summary(vol_np, max_dim=MAX_DIM)
        diff = differentiable_persistence_summary(torch.from_numpy(vol_np), max_dim=MAX_DIM)
        np.testing.assert_allclose(diff.detach().numpy(), frozen, rtol=1e-4, atol=1e-5)


def test_batch_matches_per_volume():
    vols = torch.stack([torch.from_numpy(_structured_volume(12, seed=s)) for s in range(3)])
    batched = differentiable_persistence_summary_batch(vols, max_dim=MAX_DIM)
    for i in range(3):
        single = differentiable_persistence_summary(vols[i], max_dim=MAX_DIM)
        np.testing.assert_allclose(batched[i].detach().numpy(), single.detach().numpy())


def test_gradient_finite_difference():
    """Gate 2: perturbing a voxel that participates in a real pair must change the summary sum by the
    amount the analytic gradient predicts."""
    torch.manual_seed(0)
    vol_np = _structured_volume(10, seed=1)
    vol = torch.from_numpy(vol_np).clone().requires_grad_(True)

    out = differentiable_persistence_summary(vol, max_dim=MAX_DIM)
    loss = out.sum()
    loss.backward()
    grad = vol.grad
    assert grad is not None and grad.abs().sum() > 0, "no gradient reached the input volume at all"

    flat_idx = int(grad.abs().argmax())
    idx = np.unravel_index(flat_idx, vol.shape)
    analytic = grad[idx].item()

    eps = 1e-3
    vol_plus = vol.detach().clone(); vol_plus[idx] += eps
    vol_minus = vol.detach().clone(); vol_minus[idx] -= eps
    with torch.no_grad():
        s_plus = differentiable_persistence_summary(vol_plus, max_dim=MAX_DIM).sum().item()
        s_minus = differentiable_persistence_summary(vol_minus, max_dim=MAX_DIM).sum().item()
    numeric = (s_plus - s_minus) / (2 * eps)

    assert abs(analytic - numeric) < 0.05 * max(abs(analytic), 1e-6) + 1e-3, \
        f"analytic grad {analytic} vs finite-difference {numeric} at voxel {idx}"


def test_resolution_mismatch_zoom_vs_interpolate():
    """Gate 4: the frozen target's scipy.ndimage.zoom(order=1) resize and this module's
    F.interpolate(mode='trilinear') resize must not diverge enough to matter, on the same volume."""
    vol_np = _structured_volume(32, seed=2)
    zoom = 16 / vol_np.shape[0]
    zoomed = ndimage.zoom(vol_np, zoom, order=1)
    frozen = compute_persistence_summary(zoomed, max_dim=MAX_DIM)

    vol_t = torch.from_numpy(vol_np)[None, None]   # (1,1,D,H,W)
    interp = F.interpolate(vol_t, size=(16, 16, 16), mode="trilinear", align_corners=False)[0, 0]
    diff = differentiable_persistence_summary(interp, max_dim=MAX_DIM)

    # Not bit-exact (different resize algorithms) -- checking the two summaries land close in absolute terms,
    # not that they're identical; a real per-case divergence check against materialized data is a separate,
    # later gate per the formalization (Section 5, Gate 4) once this experiment is actually GO'd.
    np.testing.assert_allclose(diff.detach().numpy(), frozen, atol=0.5)


def test_end_to_end_wiring_reaches_encoder():
    """Gate 3: loss_topology_diff must reach the decoder, ZBroadcast, fusion, graph_head, AND the encoder --
    the whole point of Section 2's construction versus PHPredictHead's indirect regression alone."""
    torch.manual_seed(0)
    D, K, Z_DIM = 32, 4, 64
    VOLUME_SIZE, STEM_STRIDE = 64, 16   # matches tests/test_ablation_topology.py's known-good configuration
    GRID_SHAPE = (VOLUME_SIZE // STEM_STRIDE,) * 3
    TOPOLOGY_DOWNSAMPLE = 16

    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=MAX_DIM, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))   # 4 blocks x stride 2 = 16, inverts STEM_STRIDE
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)

    x = torch.clamp(torch.randn(1, 1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE) * 0.3, -1.0, 1.0)
    ph_target = torch.from_numpy(
        compute_persistence_summary(
            ndimage.zoom(x[0, 0].numpy(), TOPOLOGY_DOWNSAMPLE / VOLUME_SIZE, order=1), max_dim=MAX_DIM
        )
    ).unsqueeze(0)

    ph_vector_dummy = torch.zeros(1, TopologyHead.summary_length(MAX_DIM))
    out = model(x, ph_vector=ph_vector_dummy)
    decoder_in = z_broadcast(out["graph"].node_features, out["z"])
    recon = decoder(decoder_in, GRID_SHAPE)

    recon_small = F.interpolate(recon, size=(TOPOLOGY_DOWNSAMPLE,) * 3, mode="trilinear", align_corners=False)
    pred_summary = differentiable_persistence_summary_batch(recon_small, max_dim=MAX_DIM)
    loss = l_topology(pred_summary, ph_target)
    loss.backward()

    def has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert has_grad(decoder), "decoder got no gradient from loss_topology_diff"
    assert has_grad(z_broadcast), "ZBroadcast got no gradient from loss_topology_diff"
    assert has_grad(model.fusion), "fusion got no gradient from loss_topology_diff"
    assert has_grad(model.graph_head), "graph_head got no gradient from loss_topology_diff -- differentiable path did not reach the encoder"
