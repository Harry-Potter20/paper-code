"""psi(X) — dirac_formalization.md Section 2. Conv3d stem reduces a volume to an NxN xN token grid,
a shallow Transformer mixes tokens, two real linear heads complexify. psi stays a (re, im) pair of real
tensors everywhere except the few sites that need genuine complex arithmetic (density_state.py) — this
vault has no prior complex-autograd code, so surface area is kept deliberately small.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from model.positional import sinusoidal_3d


@dataclass
class ComplexState:
    re: torch.Tensor   # (B, N, d)
    im: torch.Tensor   # (B, N, d)

    def as_complex(self) -> torch.Tensor:
        return torch.complex(self.re, self.im)

    def norm(self) -> torch.Tensor:
        """||psi_i|| per token — the magnitude used by the Section 8 magnitude-conflation check."""
        return torch.sqrt(self.re.pow(2).sum(-1) + self.im.pow(2).sum(-1) + 1e-12)


class ConvStem3D(nn.Module):
    """4 stride-2 Conv3d blocks: volume -> (volume/16)^3 token grid. e.g. 128^3 -> 8^3 = 512 tokens."""

    def __init__(self, in_channels: int = 1, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = [in_channels, *widths]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, chans[i + 1]), chans[i + 1]),
                nn.GELU(),
            )
            for i in range(len(widths))
        ])
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x   # (B, C, d', h', w')


class TokenTransformer(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)   # (B, N, dim)


class ComplexEncoder(nn.Module):
    """X -> psi. `d` is the per-token embedding width (formalization Section 1, [CALIBRATE], default 128)."""

    def __init__(self, d: int = 128, stem_widths=(32, 64, 128, 256), transformer_depth: int = 2):
        super().__init__()
        self.stem = ConvStem3D(in_channels=1, widths=stem_widths)
        self.proj_in = nn.Linear(self.stem.out_channels, d)
        self.transformer = TokenTransformer(dim=d, depth=transformer_depth)
        self.head_re = nn.Linear(d, d)
        self.head_im = nn.Linear(d, d)
        self.d = d

    def encode_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """x: (B, 1, D, H, W) -> (B, N, d) shared real hidden state, before complexification. Exposed
        separately so DiracModel's D_classical ablation (formalization Section 9) can reuse the shared
        stem+transformer without going through the complex heads at all."""
        feat = self.stem(x)                                  # (B, C, d', h', w')
        b, c, dz, hy, wx = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)              # (B, N, C), N = dz*hy*wx
        tokens = self.proj_in(tokens)                          # (B, N, d)
        pos = sinusoidal_3d((dz, hy, wx), self.d, device=tokens.device)   # (N, d), fixed, 0 params
        h = self.transformer(tokens + pos.unsqueeze(0))          # (B, N, d)
        return h, (dz, hy, wx)

    def heads(self, h: torch.Tensor) -> ComplexState:
        return ComplexState(re=self.head_re(h), im=self.head_im(h))

    def forward(self, x: torch.Tensor) -> ComplexState:
        """x: (B, 1, D, H, W), D=H=W divisible by 16."""
        h, _ = self.encode_hidden(x)
        return self.heads(h)
