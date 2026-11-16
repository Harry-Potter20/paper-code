"""Top-level DiracModel — combines psi/rho/G/T/z per dirac_formalization.md Section 6. `classical_ablation`
strips the complex/probabilistic/graph/topology branches (spec Section 19's D_classical), reusing only the
shared stem+transformer. This flag path is NOT the same thing as the parameter-matched CNN/ViT baselines in
baselines/ — see formalization Section 9 for why conflating the two would be a spec violation.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.complex_encoder import ComplexEncoder
from model.density_state import DensityHead
from model.graph import GraphHead
from model.topology import TopologyHead


class DiracModel(nn.Module):
    def __init__(self, d: int = 128, K: int = 8, grid_shape: tuple[int, int, int] = (8, 8, 8),
                top_k: int = 8, connectivity: int = 6, topology_max_dim: int = 2, z_dim: int = 256):
        super().__init__()
        self.d = d
        self.K = K
        self.grid_shape = grid_shape
        self.encoder = ComplexEncoder(d=d)
        self.density_head = DensityHead(d=d, K=K)
        self.graph_head = GraphHead(d=d, grid_shape=grid_shape, top_k=top_k, connectivity=connectivity)
        self.topology_head = TopologyHead(m=TopologyHead.summary_length(topology_max_dim), out_dim=d)
        from model.fusion import FusionHead
        self.fusion = FusionHead(d=d, K=K, topology_embed_dim=d, z_dim=z_dim)
        self.classical_pool = nn.Sequential(nn.Linear(d, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim))

    def forward(self, x: torch.Tensor, ph_vector: torch.Tensor | None = None,
               classical_ablation: bool = False) -> dict:
        """x: (B, 1, D, H, W). ph_vector: (B, m) precomputed outside autograd via
        model.topology.compute_persistence_summary — required unless classical_ablation=True."""
        h, grid_shape = self.encoder.encode_hidden(x)

        if classical_ablation:
            z = self.classical_pool(h.mean(dim=1))
            return {"z": z, "psi": None, "rho": None, "graph": None, "topology": None}

        psi = self.encoder.heads(h)
        rho = self.density_head(psi)
        graph = self.graph_head(psi)
        assert ph_vector is not None, "ph_vector required when classical_ablation=False"
        topology_embed = self.topology_head(ph_vector)
        z = self.fusion(rho, graph, topology_embed)
        return {"z": z, "psi": psi, "rho": rho, "graph": graph, "topology": topology_embed}
