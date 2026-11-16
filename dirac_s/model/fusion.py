"""z = f(rho, G, T) — dirac_formalization.md Section 6. Pools rho's per-token entropy/mixture-weight
summary, G's post-GCN node embeddings, and T's embedded PH-summary vector, then fuses through a small MLP.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.density_state import DensityState
from model.graph import RelationalGraph


class FusionHead(nn.Module):
    def __init__(self, d: int, K: int, topology_embed_dim: int, z_dim: int = 256):
        super().__init__()
        rho_summary_dim = 1 + K            # mean entropy + mean mixture-weight distribution
        graph_summary_dim = d               # mean-pooled node features
        in_dim = rho_summary_dim + graph_summary_dim + topology_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim),
        )

    def forward(self, rho: DensityState, graph: RelationalGraph, topology_embed: torch.Tensor) -> torch.Tensor:
        entropy = rho.entropy().mean(dim=-1, keepdim=True)          # (B, 1)
        weight_summary = rho.weights.mean(dim=1)                      # (B, K)
        graph_summary = graph.node_features.mean(dim=1)                # (B, d)
        combined = torch.cat([entropy, weight_summary, graph_summary, topology_embed], dim=-1)
        return self.mlp(combined)   # (B, z_dim)
