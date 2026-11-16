"""Parameter-matched 3D CNN baseline — spec Section 17. An independent architecture with its own standard
training recipe, matched to DIRAC only on parameter count/compute budget (Section 18) — NOT the
D_classical ablation in model/dirac_model.py, which shares DIRAC's own stem instead. `width` is the single
tuning knob for parameter matching (formalization Section 9: match by width, not depth).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNN3DBaseline(nn.Module):
    def __init__(self, width: int = 36, depth: int = 4, z_dim: int = 256):
        # width=36 calibrated in Phase 0b to land within 5% of N_DIRAC-S (configs/param_ledger.json)
        super().__init__()
        chans = [1] + [width * (2 ** i) for i in range(depth)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(chans[i + 1]),
                nn.GELU(),
            )
            for i in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(chans[-1], z_dim)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, D, H, W) -> (B, C, dz, hy, wx) pre-pool spatial features, C=chans[-1]. Exposed
        separately (mirrors ComplexEncoder's encode_hidden/heads split) so Phase 4's pretraining job can
        decode from real spatial structure — forward()'s AdaptiveAvgPool3d discards it, and a single pooled
        (B, z_dim) vector can't be reconstructed back into a volume."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_hidden(x)
        x = self.pool(x).flatten(1)
        return self.head(x)   # (B, z_dim)
