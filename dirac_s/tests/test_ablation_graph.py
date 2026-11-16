"""Phase 5 ablations: CPU-only, synthetic-data checks for D_-graph -- same discipline as every other
training job in this project.
"""
from __future__ import annotations
import torch

from model.dirac_model import DiracModel
from model.losses import Decoder3D, PHPredictHead, ZBroadcast, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_struct, l_topology, l_dirac
from jobs.train_ablation_graph_job import (
    encode_no_graph_mixing, sample_structure_transform, apply_structure_transform,
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


def test_node_features_equal_raw_psi_re_and_shapes_correct():
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    x = synthetic_volume()
    ph_vector = synthetic_ph_vector()
    out = encode_no_graph_mixing(model, x, ph_vector)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert torch.equal(out["graph"].node_features, out["psi"].re), \
        "node_features must be exactly raw psi.re -- no GCN mixing applied"
    assert out["graph"].adjacency.shape == (BATCH, n, n)
    assert out["rho"] is not None
    assert out["z"].shape == (BATCH, Z_DIM)


def test_graph_head_gcn_receives_no_gradient():
    """The whole point of this ablation: GraphHead's GCN must never participate in the loss at all -- not
    zeroed-out contribution, genuinely never called. Confirm its parameters get no gradient whatsoever."""
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    x = synthetic_volume()
    ph_vector = synthetic_ph_vector()

    out = encode_no_graph_mixing(model, x, ph_vector)
    decoder_in = z_broadcast(out["graph"].node_features, out["z"])
    recon = decoder(decoder_in, out["grid_shape"])
    loss = l_recon(recon, x)
    loss.backward()

    grads = [p.grad for p in model.graph_head.gcn.parameters()]
    assert all(g is None or g.abs().sum() == 0 for g in grads), "graph_head.gcn received nonzero gradient"

    # the shared encoder heads, by contrast, must receive real gradient (only graph mixing is ablated)
    head_grads = [p.grad for p in model.encoder.head_re.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in head_grads), "head_re got no gradient at all"


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
        out_x = encode_no_graph_mixing(model, x, ph_vector)
        decoder_in = z_broadcast(out_x["graph"].node_features, out_x["z"])
        recon = decoder(decoder_in, out_x["grid_shape"])
        loss_recon = l_recon(recon, x)

        token_mask = generate_block_mask(out_x["grid_shape"], mask_ratio=0.75, batch=x.shape[0])
        voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
        masked_x = x.masked_fill(voxel_mask, 0.0)
        out_m = encode_no_graph_mixing(model, masked_x, ph_vector)
        decoder_in_m = z_broadcast(out_m["graph"].node_features, out_m["z"])
        recon_m = decoder(decoder_in_m, out_m["grid_shape"])
        loss_masked = l_masked(recon_m, x, voxel_mask)

        shift_tokens, jitter_scale, jitter_offset = sample_structure_transform()
        x_g = apply_structure_transform(x, shift_tokens, jitter_scale, jitter_offset)
        h_g, _ = model.encoder.encode_hidden(x_g)
        psi_g = model.encoder.heads(h_g)
        rho_g = model.density_head(psi_g)
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
