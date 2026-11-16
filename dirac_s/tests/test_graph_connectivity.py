"""Gate 0 (IMPLEMENTATION_SPEC.md Section 1, row 0): G's adjacency has the expected shape and every node
is reachable — no orphan components on the synthetic grid (formalization Section 12, falsification gate 4).
"""
from tests._common import build_model, synthetic_batch, BATCH, GRID_SHAPE


def test_adjacency_shape():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    assert out["graph"].adjacency.shape == (BATCH, n, n)
    assert out["graph"].node_features.shape[:2] == (BATCH, n)


def test_no_orphan_nodes():
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    assert out["graph"].orphan_count() == 0


def test_graph_is_fully_connected():
    """The real reachability check (BFS-based), not just a per-node degree proxy — the lattice edges
    alone already guarantee full connectivity on a regular grid, but this test verifies it directly
    rather than trusting the construction's intent."""
    model = build_model()
    x, ph_vectors = synthetic_batch()
    out = model(x, ph_vector=ph_vectors)
    assert out["graph"].num_connected_components() == 1


def test_lattice_alone_is_connected():
    """Isolates the claim that lattice connectivity (independent of the learned top-k edges) already
    guarantees connectivity by construction — a degenerate top-k head should never be able to break this."""
    import torch
    from model.graph import lattice_adjacency, RelationalGraph

    adj = lattice_adjacency(GRID_SHAPE, connectivity=6).unsqueeze(0)
    g = RelationalGraph(adjacency=adj, node_features=torch.zeros(1, adj.shape[-1], 1))
    assert g.num_connected_components() == 1
