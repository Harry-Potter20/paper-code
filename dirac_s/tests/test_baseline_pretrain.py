"""Phase 4 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for the two baseline pretraining jobs
-- same "cheap CPU probe before GPU launches" discipline as tests/test_pretrain_losses.py. Uses the same
64^3/4x4x4-grid synthetic scale as tests/_common.py, at each baseline's own (Phase-0b-locked) width.
"""
from __future__ import annotations
import torch

from baselines.cnn3d import CNN3DBaseline
from baselines.vit3d import ViT3DBaseline
from model.losses import Decoder3D, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_dirac

BATCH = 2
VOLUME_SIZE = 64
STEM_STRIDE = 16
GRID_SHAPE = (VOLUME_SIZE // STEM_STRIDE,) * 3   # (4,4,4), matches tests/_common.py's convention
CNN_WIDTH, CNN_DEPTH = 36, 4
CNN_OUT_CHANNELS = CNN_WIDTH * (2 ** (CNN_DEPTH - 1))
VIT_WIDTH = 136


def synthetic_volume():
    torch.manual_seed(0)
    return torch.randn(BATCH, 1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)


def test_cnn_encode_hidden_shape():
    model = CNN3DBaseline(width=CNN_WIDTH, depth=CNN_DEPTH)
    x = synthetic_volume()
    feat = model.encode_hidden(x)
    assert feat.shape == (BATCH, CNN_OUT_CHANNELS, *GRID_SHAPE)
    # forward() must still work unchanged (reuses encode_hidden internally now)
    out = model(x)
    assert out.shape == (BATCH, 256)


def test_vit_encode_hidden_shape():
    model = ViT3DBaseline(width=VIT_WIDTH, patch=STEM_STRIDE, volume_size=VOLUME_SIZE)
    x = synthetic_volume()
    tokens, grid_shape = model.encode_hidden(x)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert tokens.shape == (BATCH, n, VIT_WIDTH)
    assert grid_shape == GRID_SHAPE
    out = model(x)
    assert out.shape == (BATCH, 256)


def test_cnn_decoder_round_trip():
    model = CNN3DBaseline(width=CNN_WIDTH, depth=CNN_DEPTH)
    decoder = Decoder3D(in_channels=CNN_OUT_CHANNELS, widths=(128, 64, 32, 1))
    x = synthetic_volume()
    feat = model.encode_hidden(x)
    tokens = feat.flatten(2).transpose(1, 2)
    recon = decoder(tokens, GRID_SHAPE)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_vit_decoder_round_trip():
    model = ViT3DBaseline(width=VIT_WIDTH, patch=STEM_STRIDE, volume_size=VOLUME_SIZE)
    decoder = Decoder3D(in_channels=VIT_WIDTH, widths=(96, 64, 32, 1))
    x = synthetic_volume()
    tokens, grid_shape = model.encode_hidden(x)
    recon = decoder(tokens, grid_shape)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def _five_step_recon_check(model, decoder, encode_fn):
    torch.manual_seed(0)
    x = synthetic_volume()
    params = list(model.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    losses = []
    for _ in range(5):
        tokens, grid_shape = encode_fn(model, x)
        recon = decoder(tokens, grid_shape)
        loss_recon = l_recon(recon, x)

        token_mask = generate_block_mask(grid_shape, mask_ratio=0.75, batch=x.shape[0])
        voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
        masked_x = x.masked_fill(voxel_mask, 0.0)
        tokens_m, _ = encode_fn(model, masked_x)
        recon_m = decoder(tokens_m, grid_shape)
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


def test_cnn_five_steps_reduce_loss():
    model = CNN3DBaseline(width=CNN_WIDTH, depth=CNN_DEPTH)
    decoder = Decoder3D(in_channels=CNN_OUT_CHANNELS, widths=(128, 64, 32, 1))

    def encode(m, x):
        feat = m.encode_hidden(x)
        return feat.flatten(2).transpose(1, 2), GRID_SHAPE

    _five_step_recon_check(model, decoder, encode)


def test_vit_five_steps_reduce_loss():
    model = ViT3DBaseline(width=VIT_WIDTH, patch=STEM_STRIDE, volume_size=VOLUME_SIZE)
    decoder = Decoder3D(in_channels=VIT_WIDTH, widths=(96, 64, 32, 1))
    _five_step_recon_check(model, decoder, lambda m, x: m.encode_hidden(x))
