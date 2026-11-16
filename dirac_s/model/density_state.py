"""rho(X) — dirac_formalization.md Section 3, Construction B (mixture-of-K rank-1 states). PSD and
trace-1 hold by construction; a naive rank-1 pure state (Construction A) is deliberately NOT implemented
as the default path here because its von Neumann entropy is identically zero (see Section 3) — it is kept
only as the D_-prob ablation via `pure_state()` below, never as DensityState's default construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from model.complex_encoder import ComplexState


@dataclass
class DensityState:
    weights: torch.Tensor   # (B, N, K) - pi_{i,k}, softmax-normalized, sums to 1 over K
    basis: ComplexState      # phi_{i,k} - re/im shape (B, N, K, d)

    def entropy(self) -> torch.Tensor:
        """Von Neumann entropy S(rho_i) via the K x K Gram-matrix trick (formalization Section 3
        efficiency note): rho_i's nonzero eigenvalues equal the eigenvalues of the K x K matrix
        M_jk = sqrt(pi_j pi_k) <phihat_j, phihat_k>, independent of the ambient dimension d."""
        phi = self.basis.as_complex()                                   # (B, N, K, d)
        phi_norm = phi / (phi.norm(dim=-1, keepdim=True) + 1e-12)        # unit-norm basis vectors
        gram = torch.einsum("bnkd,bnjd->bnkj", phi_norm, phi_norm.conj())  # (B, N, K, K)
        w = self.weights.clamp_min(1e-12)
        sqrt_w = torch.sqrt(w)
        M = sqrt_w.unsqueeze(-1) * gram * sqrt_w.unsqueeze(-2)             # (B, N, K, K), Hermitian PSD
        eigvals = torch.linalg.eigvalsh(M).clamp_min(0.0)                  # eigvalsh handles complex Hermitian input directly
        return -(eigvals * torch.log(eigvals + 1e-12)).sum(-1)             # (B, N)

    def to_dense(self) -> torch.Tensor:
        """Materializes the actual (B, N, d, d) complex rho from its definition — NOT the hot path
        (see entropy()'s efficient alternative); exists specifically so Gate 0 can independently verify
        PSD/trace-1 against a computational path different from entropy()'s, per IMPLEMENTATION_SPEC
        Section 0 rule 7 ("no eyeballing shapes")."""
        phi = self.basis.as_complex()                                    # (B, N, K, d)
        proj = torch.einsum("bnkd,bnke->bnkde", phi, phi.conj())          # (B, N, K, d, d), rank-1 each
        norm2 = phi.norm(dim=-1).pow(2).clamp_min(1e-12)                    # (B, N, K)
        proj = proj / norm2[..., None, None]
        w = self.weights.to(proj.dtype)
        rho = (w[..., None, None] * proj).sum(dim=2)                        # (B, N, d, d)
        return rho

    def check_psd_trace1(self, eig_tol: float = -1e-6, trace_tol: float = 1e-5) -> dict:
        """Gate 0 / Gate 2's correctness check. Materializes the dense rho (small d expected at this
        scale) and checks eigenvalues/trace directly — independent of entropy()'s K x K shortcut."""
        rho = self.to_dense()
        eigvals = torch.linalg.eigvalsh(rho)                                 # (B, N, d), rho is Hermitian
        min_eig = eigvals.min().item()
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real           # (B, N)
        max_trace_err = (trace - 1.0).abs().max().item()
        return {
            "min_eig": min_eig,
            "max_trace_error": max_trace_err,
            "psd_ok": min_eig >= eig_tol,
            "trace_ok": max_trace_err <= trace_tol,
        }


def pure_state(psi: ComplexState) -> "DensityState":
    """Construction A (formalization Section 3) — the rejected rank-1 pure state rho_i = psi_i psi_i^dag
    / ||psi_i||^2. Never used as DensityState's default; exists only so the D_-prob ablation (spec Section
    19) has a concrete, ready-to-run ρ construction to compare against, without silently re-deriving it
    later. Its entropy is identically zero for every token, by construction — that IS the ablation result."""
    weights = torch.ones(*psi.re.shape[:-1], 1, device=psi.re.device, dtype=psi.re.dtype)   # (B, N, 1)
    basis = ComplexState(re=psi.re.unsqueeze(2), im=psi.im.unsqueeze(2))                     # (B, N, 1, d)
    return DensityState(weights=weights, basis=basis)


def pure_state_padded(psi: ComplexState, K: int) -> "DensityState":
    """pure_state() above, zero-padded from K=1 to the real K so FusionHead/PHPredictHead/entropy()/
    to_dense() all stay byte-for-byte unmodified for the D_-prob ablation (jobs/train_ablation_prob_job.py)
    -- a zero-weight, zero-basis padded component contributes EXACTLY zero everywhere, not just
    numerically negligibly (verified in tests/test_ablation_prob.py): a zero basis vector normalizes to
    exactly 0 (0/eps == 0 in IEEE float), which zeroes every Gram-matrix entry and every projector touching
    that component exactly, leaving entropy()'s eigendecomposition an exact diag(1,0,...,0) and to_dense()'s
    weighted sum collapsing to exactly pure_state()'s own rank-1 projector."""
    b, n, d = psi.re.shape
    weights = torch.zeros(b, n, K, device=psi.re.device, dtype=psi.re.dtype)
    weights[..., 0] = 1.0
    re = torch.zeros(b, n, K, d, device=psi.re.device, dtype=psi.re.dtype)
    im = torch.zeros(b, n, K, d, device=psi.im.device, dtype=psi.im.dtype)
    re[:, :, 0, :] = psi.re
    im[:, :, 0, :] = psi.im
    return DensityState(weights=weights, basis=ComplexState(re=re, im=im))


class DensityHead(nn.Module):
    """psi -> DensityState. K basis projections W_k (as real re/im Linear pairs, never a complex-dtype
    layer — keeps the parameter-counting convention in formalization Section 9 automatically correct,
    since two real Linears of size d_in x d_out already sum to 2*d_in*d_out real scalars)."""

    def __init__(self, d: int, K: int = 8):
        super().__init__()
        self.K = K
        self.mixture_logits = nn.Linear(2 * d, K)     # g_theta(psi_i), from concatenated (re, im)
        self.basis_re = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])
        self.basis_im = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])

    def forward(self, psi: ComplexState) -> DensityState:
        cat = torch.cat([psi.re, psi.im], dim=-1)               # (B, N, 2d)
        weights = torch.softmax(self.mixture_logits(cat), dim=-1)   # (B, N, K)
        # phi_k = W_k(psi): a genuine complex linear map applied to a complex input, realized via the
        # standard real/imag decomposition (W_re+iW_im)(x_re+ix_im) = (W_re x_re - W_im x_im) + i(W_re x_im + W_im x_re).
        re_parts, im_parts = [], []
        for k in range(self.K):
            wre, wim = self.basis_re[k], self.basis_im[k]
            re_parts.append(wre(psi.re) - wim(psi.im))
            im_parts.append(wre(psi.im) + wim(psi.re))
        basis = ComplexState(re=torch.stack(re_parts, dim=2), im=torch.stack(im_parts, dim=2))  # (B,N,K,d)
        return DensityState(weights=weights, basis=basis)
