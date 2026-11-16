"""Parameter-matched 3D ViT baseline — spec Section 17. Independent architecture, own recipe, matched to
DIRAC only on parameter count/compute budget (Section 18). `width` (the token embedding dim) is the
tuning knob for parameter matching, depth held fixed (formalization Section 9: match by width, not depth).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.positional import sinusoidal_3d


class ViT3DBaseline(nn.Module):
    def __init__(self, width: int = 136, depth: int = 4, heads: int = 4, patch: int = 16,
                volume_size: int = 128, z_dim: int = 256):
        super().__init__()
        # width=136 calibrated in Phase 0b to land within 5% of N_DIRAC-S (configs/param_ledger.json).
        # Positional encoding is fixed (sinusoidal_3d), not a learned nn.Parameter sized to n_patches —
        # keeps this calibration valid independent of volume_size (see model/positional.py docstring).
        assert volume_size % patch == 0
        self.patch_embed = nn.Conv3d(1, width, kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(width, z_dim)
        self.width = width

    def encode_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """x: (B, 1, D, H, W) -> (B, n_patches, width) pre-pool token sequence + grid shape. Exposed
        separately (mirrors ComplexEncoder's encode_hidden/heads split) so Phase 4's pretraining job can
        decode from real per-token structure — forward()'s mean-pool discards it."""
        feat = self.patch_embed(x)                                        # (B, width, gz, gy, gx)
        gz, gy, gx = feat.shape[2:]
        tokens = feat.flatten(2).transpose(1, 2)                            # (B, n_patches, width)
        pos = sinusoidal_3d((gz, gy, gx), self.width, device=tokens.device)
        h = self.encoder(tokens + pos.unsqueeze(0))
        return h, (gz, gy, gx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encode_hidden(x)
        return self.head(h.mean(dim=1))   # (B, z_dim)
