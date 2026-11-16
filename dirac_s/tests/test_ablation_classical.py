"""Phase 5 ablations: CPU-only, synthetic-data checks for D_classical -- same discipline as every other
training job in this project.
"""
from __future__ import annotations
import torch

from model.dirac_model import DiracModel
from model.losses import Decoder3D, ZBroadcast, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_dirac
from jobs.train_ablation_classical_job import encode_classical, forward_recon

BATCH = 2
VOLUME_SIZE = 64
STEM_STRIDE = 16
GRID_SHAPE = (VOLUME_SIZE // STEM_STRIDE,) * 3
D, K, Z_DIM = 64, 8, 256


def synthetic_volume():
    torch.manual_seed(0)
    return torch.clamp(torch.randn(BATCH, 1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE) * 0.3, -1.0, 1.0)


def test_classical_ablation_returns_none_structural_branches():
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    x = synthetic_volume()
    out = model(x, classical_ablation=True)
    assert out["psi"] is None and out["rho"] is None and out["graph"] is None and out["topology"] is None
    assert out["z"].shape == (BATCH, Z_DIM)


def test_encode_classical_shapes():
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    x = synthetic_volume()
    h, z, grid_shape = encode_classical(model, x)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert h.shape == (BATCH, n, D)
    assert z.shape == (BATCH, Z_DIM)
    assert grid_shape == GRID_SHAPE


def test_forward_recon_round_trip_shape():
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    x = synthetic_volume()
    recon, grid_shape = forward_recon(model, decoder, z_broadcast, x)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_classical_pool_receives_gradient_via_z_broadcast():
    """The whole point of decoding from z_broadcast(h, z) instead of h alone: classical_pool must actually
    train, not sit at random init. Confirm its parameters get a nonzero gradient from the reconstruction loss."""
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    x = synthetic_volume()

    recon, _ = forward_recon(model, decoder, z_broadcast, x)
    loss = l_recon(recon, x)
    loss.backward()

    grads = [p.grad for p in model.classical_pool.parameters()]
    assert all(g is not None for g in grads), "classical_pool got no gradient at all"
    assert any(g.abs().sum() > 0 for g in grads), "classical_pool's gradient is all zero"


def test_five_optimizer_steps_reduce_loss_on_fixed_synthetic_batch():
    torch.manual_seed(0)
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    params = list(model.parameters()) + list(decoder.parameters()) + list(z_broadcast.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = synthetic_volume()

    losses = []
    for _ in range(5):
        recon, grid_shape = forward_recon(model, decoder, z_broadcast, x)
        loss_recon = l_recon(recon, x)

        token_mask = generate_block_mask(grid_shape, mask_ratio=0.75, batch=x.shape[0])
        voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
        masked_x = x.masked_fill(voxel_mask, 0.0)
        recon_m, _ = forward_recon(model, decoder, z_broadcast, masked_x)
        loss_masked = l_masked(recon_m, x, voxel_mask)

        zero = loss_recon.new_zeros(())
        total = l_dirac(loss_recon, loss_masked, zero, zero, lam_s=0.0, lam_t=0.0)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        losses.append(total.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], f"loss did not decrease over 5 steps: {losses}"
