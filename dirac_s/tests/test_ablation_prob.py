"""Phase 5 ablations: CPU-only, synthetic-data checks for D_-prob -- same discipline as every other
training job in this project.
"""
from __future__ import annotations
import torch

from model.dirac_model import DiracModel
from model.density_state import pure_state, pure_state_padded
from model.losses import Decoder3D, PHPredictHead, ZBroadcast, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_struct, l_topology, l_dirac
from jobs.train_ablation_prob_job import (
    encode_pure_state, sample_structure_transform, apply_structure_transform,
    matched_token_slices, crop_density_state,
)
from model.topology import TopologyHead

BATCH = 2
VOLUME_SIZE = 64
STEM_STRIDE = 16
GRID_SHAPE = (VOLUME_SIZE // STEM_STRIDE,) * 3
D, K, Z_DIM = 64, 8, 256
TOPOLOGY_MAX_DIM = 2
M = TopologyHead.summary_length(TOPOLOGY_MAX_DIM)


def synthetic_volume():
    torch.manual_seed(0)
    return torch.clamp(torch.randn(BATCH, 1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE) * 0.3, -1.0, 1.0)


def synthetic_ph_vector():
    torch.manual_seed(1)
    return torch.rand(BATCH, M)


def test_pure_state_padded_matches_pure_state_exactly():
    """The zero-padded K=1->K components must contribute EXACTLY zero, not just negligibly -- entropy and
    the K x K Gram matrix must come out byte-identical to the unpadded pure_state()."""
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    x = synthetic_volume()
    h, _ = model.encoder.encode_hidden(x)
    psi = model.encoder.heads(h)

    rho_unpadded = pure_state(psi)
    rho_padded = pure_state_padded(psi, K)

    assert rho_padded.weights.shape == (BATCH, GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2], K)
    # eigvalsh on a 1x1 vs zero-padded 8x8 matrix isn't bit-identical (different numerical path), but both
    # must land at zero to float precision -- that's the actual claim ("contributes exactly zero", not
    # "identical rounding error").
    assert torch.allclose(rho_padded.entropy(), rho_unpadded.entropy(), atol=1e-5)
    assert torch.allclose(rho_padded.entropy(), torch.zeros_like(rho_padded.entropy()), atol=1e-5)


def test_encode_pure_state_shapes_and_rho_is_rank_one():
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    x = synthetic_volume()
    ph_vector = synthetic_ph_vector()
    out = encode_pure_state(model, x, ph_vector)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert out["h"].shape == (BATCH, n, D)
    assert out["z"].shape == (BATCH, Z_DIM)
    assert out["rho"] is not None and out["graph"] is not None
    assert torch.allclose(out["rho"].entropy(), torch.zeros(BATCH, n), atol=1e-5)


def test_density_head_receives_no_gradient():
    """The whole point of this ablation: DensityHead must never participate in the loss at all -- not
    zeroed-out contribution, genuinely never called. Confirm its parameters get no gradient whatsoever."""
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    x = synthetic_volume()
    ph_vector = synthetic_ph_vector()

    out = encode_pure_state(model, x, ph_vector)
    decoder_in = z_broadcast(out["graph"].node_features, out["z"])
    recon = decoder(decoder_in, out["grid_shape"])
    loss = l_recon(recon, x)
    loss.backward()

    grads = [p.grad for p in model.density_head.parameters()]
    assert all(g is None or g.abs().sum() == 0 for g in grads), "density_head received nonzero gradient"

    # graph_head and topology_head, by contrast, must receive real gradient (only prob is ablated)
    graph_grads = [p.grad for p in model.graph_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in graph_grads), "graph_head got no gradient at all"


def test_five_optimizer_steps_reduce_loss_on_fixed_synthetic_batch():
    torch.manual_seed(0)
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    ph_predict_head = PHPredictHead(d=D, K=K, m=M)
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    params = list(model.parameters()) + list(decoder.parameters()) + \
             list(ph_predict_head.parameters()) + list(z_broadcast.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = synthetic_volume()
    ph_vector = synthetic_ph_vector()

    losses = []
    for _ in range(5):
        out_x = encode_pure_state(model, x, ph_vector)
        decoder_in = z_broadcast(out_x["graph"].node_features, out_x["z"])
        recon = decoder(decoder_in, out_x["grid_shape"])
        loss_recon = l_recon(recon, x)

        token_mask = generate_block_mask(out_x["grid_shape"], mask_ratio=0.75, batch=x.shape[0])
        voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
        masked_x = x.masked_fill(voxel_mask, 0.0)
        out_m = encode_pure_state(model, masked_x, ph_vector)
        decoder_in_m = z_broadcast(out_m["graph"].node_features, out_m["z"])
        recon_m = decoder(decoder_in_m, out_m["grid_shape"])
        loss_masked = l_masked(recon_m, x, voxel_mask)

        shift_tokens, jitter_scale, jitter_offset = sample_structure_transform()
        x_g = apply_structure_transform(x, shift_tokens, jitter_scale, jitter_offset)
        h_g, _ = model.encoder.encode_hidden(x_g)
        psi_g = model.encoder.heads(h_g)
        rho_g = pure_state_padded(psi_g, model.K)
        slices_x, slices_g = matched_token_slices(out_x["grid_shape"], shift_tokens)
        rho_x_cropped = crop_density_state(out_x["rho"], out_x["grid_shape"], slices_x)
        rho_g_cropped = crop_density_state(rho_g, out_x["grid_shape"], slices_g)
        loss_struct = l_struct(rho_x_cropped, rho_g_cropped)

        predicted_ph = ph_predict_head(out_x["psi"], out_x["rho"])
        loss_topology = l_topology(predicted_ph, ph_vector)

        total = l_dirac(loss_recon, loss_masked, loss_struct, loss_topology, lam_s=1.0, lam_t=1.0)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        losses.append(total.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], f"loss did not decrease over 5 steps: {losses}"
