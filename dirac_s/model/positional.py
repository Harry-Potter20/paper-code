"""Fixed (non-learned) 3D sinusoidal positional encoding, shared by ComplexEncoder and the ViT3D
baseline. Two reasons it's fixed rather than a learned nn.Parameter: (1) a Transformer over flattened
tokens has no spatial signal at all without one — a real correctness gap, not a nice-to-have, for a model
whose entire premise is spatial/relational/topological structure; (2) a learned positional embedding sized
to a specific grid_shape/n_patches makes parameter count depend on input volume size, which would silently
invalidate the Section 9 parameter-matching calibration the moment Phase 1 moves from Gate 0's synthetic
64^3 input to the real 128^3 target. A fixed encoding contributes zero trainable parameters either way.
"""
from __future__ import annotations
import torch


def sinusoidal_3d(grid_shape: tuple[int, int, int], dim: int, device=None) -> torch.Tensor:
    """(N, dim) encoding, N = prod(grid_shape). dim split three ways across (z, y, x); each third uses
    the standard sin/cos frequency ladder. Robust to any dim >= 6 (rounds each axis's share up to an even
    number internally, then truncates the concatenation back to exactly `dim`) — deliberately not
    requiring dim % 6 == 0, since dim is a parameter-count-calibrated value (Section 9) that must not be
    re-chosen just to satisfy a positional-encoding implementation detail."""
    dz, hy, wx = grid_shape
    assert dim >= 6, f"positional dim too small for a 3-axis sin/cos encoding, got {dim}"
    third = -(-dim // 3)            # ceil(dim / 3)
    third += third % 2               # round up to even, for sin/cos pairing

    def axis_encoding(length: int) -> torch.Tensor:
        pos = torch.arange(length, device=device).float().unsqueeze(1)         # (length, 1)
        freq = torch.exp(torch.arange(0, third, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / third))
        enc = torch.zeros(length, third, device=device)
        enc[:, 0::2] = torch.sin(pos * freq)
        enc[:, 1::2] = torch.cos(pos * freq)
        return enc

    ez, ey, ex = axis_encoding(dz), axis_encoding(hy), axis_encoding(wx)
    grid = torch.zeros(dz, hy, wx, 3 * third, device=device)
    grid[..., 0:third] = ez.view(dz, 1, 1, third)
    grid[..., third:2 * third] = ey.view(1, hy, 1, third)
    grid[..., 2 * third:3 * third] = ex.view(1, 1, wx, third)
    return grid.view(dz * hy * wx, 3 * third)[:, :dim]
