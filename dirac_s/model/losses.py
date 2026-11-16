"""L_recon, L_masked, L_struct, L_topology, L_DIRAC — dirac_formalization.md Section 7. Phase 0b:
correct shapes/signatures on synthetic data. Phase 3: wired into the real pretraining job with masking
integrated into the encoder and real gudhi targets computed per batch (outside autograd).

Decoder3D, PHPredictHead and ZBroadcast are deliberately pretraining-only scaffolding, instantiated by
jobs/pretrain_dirac_job.py alongside a DiracModel instance, never as DiracModel submodules — same
convention as an MAE decoder or a SimCLR projection head, discarded once pretraining ends. Formalization
Section 6: "z is the frozen representation evaluated in Stage II probing" — the decoder and PH-prediction
head never appear at probing/eval time, so folding them into DiracModel would silently grow N_DIRAC-S past
the count Gate 0 already locked (configs/param_ledger.json) for no benefit.

Section 7's four objectives, read literally, only ever touch h/psi/rho/ph_vector — none of them depend on
GraphHead's node_features or FusionHead's z. Decoding straight from h (the most literal reading) would leave
G and the fusion MLP randomly initialized through all of Phase 3, which would quietly undermine Phase 5's
H2/H3 (graph, topology) results — a null H2/H3 would be indistinguishable from "the module was never
trained." ZBroadcast exists to close that gap: the reconstruction target is graph.node_features (post-GCN,
so G gets a real gradient) with z broadcast back into every token (so rho->fusion and topology_head's
embedding MLP get one too) — every structural branch is actually trained by pretraining, matching Section
6's framing of z as the trained representation, not §7's literal (and, on inspection, incomplete) wiring.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.complex_encoder import ComplexState
from model.density_state import DensityState


class Decoder3D(nn.Module):
    """Mirrors ConvStem3D in reverse: (B, C, d', h', w') token grid -> (B, 1, D, H, W) volume."""

    def __init__(self, in_channels: int = 128, widths=(128, 64, 32, 1)):
        super().__init__()
        chans = [in_channels, *widths]
        blocks = []
        for i in range(len(widths)):
            is_last = i == len(widths) - 1
            blocks.append(nn.Sequential(
                nn.ConvTranspose3d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                *([] if is_last else [nn.GroupNorm(min(8, chans[i + 1]), chans[i + 1]), nn.GELU()]),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, tokens: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
        """tokens: (B, N, C) -> (B, C, dz, hy, wx) -> upsample to (B, 1, D, H, W)."""
        b, n, c = tokens.shape
        dz, hy, wx = grid_shape
        x = tokens.transpose(1, 2).reshape(b, c, dz, hy, wx)
        for block in self.blocks:
            x = block(x)
        return x


class ZBroadcast(nn.Module):
    """Projects z (B, z_dim) -> (B, d) and broadcasts it onto every one of graph.node_features' N tokens,
    forming the decoder's actual input: decoder_input = graph.node_features + broadcast(z). This is what
    gives GraphHead, FusionHead and TopologyHead's embedding MLP a real gradient path from L_recon/L_masked
    (see module docstring) — without it, decoding from raw h would train only the encoder/density branches."""

    def __init__(self, z_dim: int, d: int):
        super().__init__()
        self.proj = nn.Linear(z_dim, d)

    def forward(self, node_features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return node_features + self.proj(z).unsqueeze(1)   # (B, N, d) + (B, 1, d)


def l_recon(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Objective A. L1 voxelwise, full volume."""
    return F.l1_loss(recon, target)


def l_masked(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Objective B. L1 only over masked-token voxels. `mask`: (B, 1, D, H, W) bool, True = masked."""
    if mask.sum() == 0:
        return recon.new_zeros(())
    return F.l1_loss(recon[mask], target[mask])


def generate_block_mask(grid_shape: tuple[int, int, int], mask_ratio: float = 0.75,
                        batch: int = 1, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Standard 3D MAE block masking (formalization Section 7 Objective B: mask 60-75% of the 512 tokens
    before the encoder). mask_ratio=0.75 is the upper/standard-MAE end of that [CALIBRATE] range — not yet
    data-calibrated, since mask ratio is a training hyperparameter Gate 1/2's dataset statistics don't bear
    on; adjustable via Gate 3's own FALLBACK path if the smoke run shows instability.
    Returns (B, dz, hy, wx) bool, True = masked, independently sampled per batch element."""
    dz, hy, wx = grid_shape
    n = dz * hy * wx
    n_mask = int(round(n * mask_ratio))
    noise = torch.rand(batch, n, device=device, generator=generator)
    mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
    idx = noise.argsort(dim=-1)[:, :n_mask]
    mask.scatter_(-1, idx, True)
    return mask.view(batch, dz, hy, wx)


def upsample_token_mask_to_voxels(token_mask: torch.Tensor, stem_stride: int = 16) -> torch.Tensor:
    """(B, dz, hy, wx) bool -> (B, 1, dz*stride, hy*stride, wx*stride) bool, by repeat — the exact inverse
    of ComplexEncoder's stride-16 conv-stem downsampling, so a masked token block maps to the precise voxel
    region that produced it. Used to zero-fill the input volume before the stem (Section 7's "mask ...
    before the encoder"), since this codebase's conv stem has no notion of dropping/skipping patches."""
    m = token_mask.unsqueeze(1).float()   # (B, 1, dz, hy, wx)
    m = m.repeat_interleave(stem_stride, dim=2).repeat_interleave(stem_stride, dim=3).repeat_interleave(stem_stride, dim=4)
    return m.bool()


def l_struct(rho_x: DensityState, rho_gx: DensityState) -> torch.Tensor:
    """Objective C. Frobenius distance between matched-token density states, per formalization Section 7
    (Frobenius chosen explicitly over quantum fidelity — no matrix square root per token per step)."""
    a, b = rho_x.to_dense(), rho_gx.to_dense()
    diff = a - b
    return (diff.conj() * diff).real.sum(dim=(-2, -1)).mean()


def l_topology(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Objective D. Regression to a stop-gradient gudhi-computed target — never backprop through gudhi
    itself (formalization Section 5/7)."""
    return F.mse_loss(predicted, target.detach())


class PHPredictHead(nn.Module):
    """Predicts PHsummary(X) from (psi, rho) — formalization Section 5: "psi/rho are trained to be
    predictive of T(X) via an auxiliary loss (Section 7, L_topology) — this is what makes topology
    actually shape the learned representation." This is NOT the same module as model/topology.py's
    TopologyHead, which embeds the already-known ground-truth ph_vector into z; that embedding is a
    direct input feature, not a prediction, so training against it would teach this head nothing. Pools
    the same (psi, rho) summary features FusionHead already computes, through a small MLP to a vector of
    length TopologyHead.summary_length(max_dim), regressed against the real ph_vector via l_topology.
    Pretraining-only scaffolding (see module docstring) — not a DiracModel submodule."""

    def __init__(self, d: int, K: int, m: int):
        super().__init__()
        in_dim = 2 * d + 1 + K   # mean-pooled (psi.re, psi.im) + mean entropy + mean mixture weights
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, m))

    def forward(self, psi: ComplexState, rho: DensityState) -> torch.Tensor:
        psi_summary = torch.cat([psi.re.mean(dim=1), psi.im.mean(dim=1)], dim=-1)   # (B, 2d)
        entropy_summary = rho.entropy().mean(dim=-1, keepdim=True)                    # (B, 1)
        weight_summary = rho.weights.mean(dim=1)                                       # (B, K)
        combined = torch.cat([psi_summary, entropy_summary, weight_summary], dim=-1)
        return self.mlp(combined)


def l_topology_stability(summary_gx: torch.Tensor, summary_x: torch.Tensor) -> torch.Tensor:
    """Gradient-free stability check for topology-preserving g — a diagnostic, not part of the trained
    loss (both inputs are precomputed, detached PH summaries)."""
    return (summary_gx.detach() - summary_x.detach()).norm(dim=-1).mean()


def l_dirac(recon_loss: torch.Tensor, masked_loss: torch.Tensor, struct_loss: torch.Tensor,
           topology_loss: torch.Tensor, lam_r: float = 1.0, lam_m: float = 1.0,
           lam_s: float = 1.0, lam_t: float = 1.0) -> torch.Tensor:
    """L_DIRAC = lam_r*L_recon + lam_m*L_masked + lam_s*L_struct + lam_t*L_topology. Coefficients
    [CALIBRATE], locked in configs/thresholds.lock.json before final test evaluation (spec Section 12)."""
    return lam_r * recon_loss + lam_m * masked_loss + lam_s * struct_loss + lam_t * topology_loss
