"""Phase 3 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for pretrain_dirac_job.py's new
pieces — run before any GPU/HF Job spend, per the standing "cheap CPU probe of API assumptions before GPU
launches" discipline. Mirrors tests/_common.py's Gate-0 synthetic fixture (64^3 volume, 4x4x4=64-token
grid) rather than the real 128^3/512-token scale, since these are shape/wiring/finiteness checks, not the
real-data checks Gate 2 already covers.
"""
from __future__ import annotations
import torch

from tests._common import build_model, synthetic_batch, BATCH, GRID_SHAPE, VOLUME_SIZE, D, K

STEM_STRIDE = VOLUME_SIZE // GRID_SHAPE[0]   # 64 // 4 = 16 — ConvStem3D's fixed 4x stride-2 downsample,
                                              # same value at this synthetic scale as at the real 128^3 one
from model.topology import TopologyHead
from model.losses import (
    Decoder3D, PHPredictHead, ZBroadcast,
    generate_block_mask, upsample_token_mask_to_voxels,
    l_recon, l_masked, l_struct, l_topology, l_dirac,
)
from jobs.pretrain_dirac_job import (
    sample_structure_transform, apply_structure_transform,
    matched_token_slices, crop_density_state, encode_full,
)

M = TopologyHead.summary_length(max_dim=2)


def test_generate_block_mask_ratio_and_shape():
    mask = generate_block_mask(GRID_SHAPE, mask_ratio=0.75, batch=BATCH)
    assert mask.shape == (BATCH, *GRID_SHAPE)
    assert mask.dtype == torch.bool
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    n_masked = mask.view(BATCH, -1).sum(-1)
    assert torch.all(n_masked == round(n * 0.75))


def test_upsample_token_mask_to_voxels_shape_and_correspondence():
    token_mask = generate_block_mask(GRID_SHAPE, mask_ratio=0.5, batch=BATCH)
    voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
    dz, hy, wx = GRID_SHAPE
    assert voxel_mask.shape == (BATCH, 1, dz * STEM_STRIDE, hy * STEM_STRIDE, wx * STEM_STRIDE)
    # a masked token's whole voxel block must be uniformly masked, and vice versa
    block = voxel_mask[0, 0, :STEM_STRIDE, :STEM_STRIDE, :STEM_STRIDE]
    assert bool(block.all()) == bool(token_mask[0, 0, 0, 0]) or bool((~block).all()) == (not token_mask[0, 0, 0, 0])


def test_ph_predict_head_output_shape():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    head = PHPredictHead(d=D, K=K, m=M)
    predicted = head(out["psi"], out["rho"])
    assert predicted.shape == (BATCH, M)
    assert torch.isfinite(predicted).all()


def test_z_broadcast_output_shape():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    zb = ZBroadcast(z_dim=out["z"].shape[-1], d=D)
    decoder_in = zb(out["graph"].node_features, out["z"])
    assert decoder_in.shape == out["graph"].node_features.shape


def test_decoder_reconstructs_volume_shape():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    zb = ZBroadcast(z_dim=out["z"].shape[-1], d=D)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    decoder_in = zb(out["graph"].node_features, out["z"])
    recon = decoder(decoder_in, GRID_SHAPE)
    assert recon.shape == x.shape


def test_structure_transform_and_matched_crop():
    x, _ = synthetic_batch()
    shift_tokens, jitter_scale, jitter_offset = sample_structure_transform()
    # force a nonzero shift on at least one axis so the boundary-crop path is actually exercised
    shift_tokens = (1, 0, 0)
    x_g = apply_structure_transform(x, shift_tokens, jitter_scale, jitter_offset)
    assert x_g.shape == x.shape
    assert torch.isfinite(x_g).all()

    model = build_model()
    _, ph_vectors = synthetic_batch()
    out_x = encode_full(model, x, ph_vectors)
    h_g, _ = model.encoder.encode_hidden(x_g)
    psi_g = model.encoder.heads(h_g)
    rho_g = model.density_head(psi_g)

    slices_x, slices_g = matched_token_slices(GRID_SHAPE, shift_tokens)
    rho_x_cropped = crop_density_state(out_x["rho"], GRID_SHAPE, slices_x)
    rho_g_cropped = crop_density_state(rho_g, GRID_SHAPE, slices_g)
    # shift_tokens=(1,0,0) on a 4-token z-axis drops exactly 1 of 4 z-layers -> 3/4 of tokens remain
    expected_n = (GRID_SHAPE[0] - 1) * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert rho_x_cropped.weights.shape == (BATCH, expected_n, K)
    assert rho_g_cropped.weights.shape == (BATCH, expected_n, K)

    loss = l_struct(rho_x_cropped, rho_g_cropped)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_l_dirac_finite_on_synthetic_data():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)

    zb = ZBroadcast(z_dim=out["z"].shape[-1], d=D)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    decoder_in = zb(out["graph"].node_features, out["z"])
    recon = decoder(decoder_in, GRID_SHAPE)
    loss_recon = l_recon(recon, x)

    token_mask = generate_block_mask(GRID_SHAPE, mask_ratio=0.75, batch=BATCH)
    voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
    loss_masked = l_masked(recon, x, voxel_mask)

    head = PHPredictHead(d=D, K=K, m=M)
    predicted_ph = head(out["psi"], out["rho"])
    loss_topology = l_topology(predicted_ph, ph_vectors)

    total = l_dirac(loss_recon, loss_masked, loss_recon.new_zeros(()), loss_topology)
    assert torch.isfinite(total)


def test_five_optimizer_steps_reduce_loss_on_fixed_synthetic_batch():
    """The cheapest possible end-to-end stability probe: does the exact loss/wiring combination Gate 3's
    smoke run will use actually reduce loss on a single fixed batch, with no real data or GPU involved."""
    torch.manual_seed(0)
    model = build_model()
    x, ph_vectors = synthetic_batch()

    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    head = PHPredictHead(d=D, K=K, m=M)
    zb = ZBroadcast(z_dim=256, d=D)   # matches DiracModel's default z_dim
    params = list(model.parameters()) + list(decoder.parameters()) + list(head.parameters()) + list(zb.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    losses = []
    for _ in range(5):
        out = model(x, ph_vector=ph_vectors)
        decoder_in = zb(out["graph"].node_features, out["z"])
        recon = decoder(decoder_in, GRID_SHAPE)
        loss_recon = l_recon(recon, x)
        predicted_ph = head(out["psi"], out["rho"])
        loss_topology = l_topology(predicted_ph, ph_vectors)
        total = l_dirac(loss_recon, loss_recon.new_zeros(()), loss_recon.new_zeros(()), loss_topology)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        losses.append(total.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], f"loss did not decrease over 5 steps: {losses}"
