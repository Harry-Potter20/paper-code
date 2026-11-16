"""G — dirac_formalization.md Section 4, Construction A (patch-adjacency). Nodes are psi's own tokens;
edges are fixed lattice connectivity plus learned top-k feature-similarity edges. A from-scratch 2-layer
GCN, not GAT/torch_geometric — see Section 4 for why (avoids a second attention mechanism confounding the
H3 ablation, avoids a compiled-CUDA-extension dependency never exercised on this vault's HF Jobs images).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from model.complex_encoder import ComplexState


@dataclass
class RelationalGraph:
    adjacency: torch.Tensor        # (B, N, N), binary/weighted, symmetric
    node_features: torch.Tensor    # (B, N, d), post-GCN

    def orphan_count(self) -> int:
        """Number of nodes with zero edges (self excluded) — Gate 0/2's connectivity check."""
        deg = self.adjacency.sum(-1) - torch.diagonal(self.adjacency, dim1=-2, dim2=-1)
        return int((deg <= 0).sum().item())

    def num_connected_components(self) -> int:
        """Real connected-components count via iterative BFS/flood-fill over the (dense) adjacency —
        the actual "every node reachable" check, not just a per-node degree proxy."""
        adj = (self.adjacency[0] > 0)   # first batch element; grid topology is shared across the batch
        n = adj.shape[0]
        visited = torch.zeros(n, dtype=torch.bool, device=adj.device)
        components = 0
        for start in range(n):
            if visited[start]:
                continue
            components += 1
            frontier = torch.zeros(n, dtype=torch.bool, device=adj.device)
            frontier[start] = True
            while frontier.any():
                visited |= frontier
                frontier = (adj & frontier.unsqueeze(0)).any(dim=1) & ~visited
        return components


def lattice_adjacency(grid_shape: tuple[int, int, int], connectivity: int = 6, device=None) -> torch.Tensor:
    """Fixed 6- or 26-connectivity adjacency over a dz x hy x wx token grid, flattened to N=dz*hy*wx."""
    dz, hy, wx = grid_shape
    n = dz * hy * wx
    idx = torch.arange(n, device=device).view(dz, hy, wx)
    adj = torch.zeros(n, n, device=device)
    offsets6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    offsets26 = [(dz_, hy_, wx_) for dz_ in (-1, 0, 1) for hy_ in (-1, 0, 1) for wx_ in (-1, 0, 1)
                 if (dz_, hy_, wx_) != (0, 0, 0)]
    offsets = offsets6 if connectivity == 6 else offsets26
    for oz, oy, ox in offsets:
        src = idx[max(0, -oz):dz - max(0, oz), max(0, -oy):hy - max(0, oy), max(0, -ox):wx - max(0, ox)]
        dst = idx[max(0, oz):dz - max(0, -oz), max(0, oy):hy - max(0, -oy), max(0, ox):wx - max(0, -ox)]
        adj[src.reshape(-1), dst.reshape(-1)] = 1.0
    return adj   # (N, N)


def add_topk_feature_edges(adj: torch.Tensor, psi: ComplexState, k: int = 8) -> torch.Tensor:
    """Adds top-k feature-similarity edges in psi-space on top of a fixed lattice adjacency — captures
    non-local correspondence (e.g. left/right anatomical symmetry) lattice adjacency alone cannot."""
    feat = torch.cat([psi.re, psi.im], dim=-1)                  # (B, N, 2d)
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
    sim = torch.einsum("bnd,bmd->bnm", feat, feat)               # (B, N, N) cosine similarity
    b, n, _ = sim.shape
    sim = sim - torch.eye(n, device=sim.device).unsqueeze(0) * 2   # exclude self
    topk_idx = sim.topk(k, dim=-1).indices                          # (B, N, k)
    edges = torch.zeros(b, n, n, device=sim.device)
    edges.scatter_(-1, topk_idx, 1.0)
    edges = torch.maximum(edges, edges.transpose(-1, -2))            # symmetrize
    return torch.clamp(adj.unsqueeze(0) + edges, max=1.0)             # (B, N, N)


class GraphConvNaive(nn.Module):
    """A plain 2-layer Kipf-and-Welling GCN: H' = sigma(D^-1/2 A D^-1/2 H W). Dense N=512 adjacency is
    cheap (~262K floats) at DIRAC-S scale — no sparse message-passing framework needed."""

    def __init__(self, d: int):
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    @staticmethod
    def _normalize(adj: torch.Tensor) -> torch.Tensor:
        adj_hat = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)   # self-loops
        deg = adj_hat.sum(-1)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)

    def forward(self, adjacency: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        a_norm = self._normalize(adjacency)
        h = torch.relu(self.w1(torch.bmm(a_norm, node_features)))
        h = torch.relu(self.w2(torch.bmm(a_norm, h)))
        return h


class GraphHead(nn.Module):
    def __init__(self, d: int, grid_shape: tuple[int, int, int], top_k: int = 8, connectivity: int = 6):
        super().__init__()
        self.grid_shape = grid_shape
        self.top_k = top_k
        self.connectivity = connectivity
        self.gcn = GraphConvNaive(d)
        self._lattice_cache: dict = {}

    def _lattice(self, device) -> torch.Tensor:
        key = (self.grid_shape, self.connectivity, device)
        if key not in self._lattice_cache:
            self._lattice_cache[key] = lattice_adjacency(self.grid_shape, self.connectivity, device=device)
        return self._lattice_cache[key]

    def forward(self, psi: ComplexState) -> RelationalGraph:
        lattice = self._lattice(psi.re.device)
        adjacency = add_topk_feature_edges(lattice, psi, k=self.top_k)
        node_features = self.gcn(adjacency, psi.re)   # GCN mixes over the real component; im carried in psi elsewhere
        return RelationalGraph(adjacency=adjacency, node_features=node_features)
