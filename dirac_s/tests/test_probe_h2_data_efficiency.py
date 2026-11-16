"""Phase 5 slice 4 (IMPLEMENTATION_SPEC.md): CPU-only, synthetic-data checks for H2's data-efficiency
curve -- same discipline as the other probe test files. Focuses on the one genuinely new piece of logic
(nested fraction subsampling); probe training/eval reuses the already-tested pattern from
test_probe_h3_topology.py.
"""
from __future__ import annotations
import numpy as np

from jobs.probe_h2_data_efficiency_job import nested_fraction_indices, trapezoidal_area, FRACTIONS


def test_trapezoidal_area_matches_known_cases():
    # constant y=1 over x in [0,1] -> area 1
    assert abs(trapezoidal_area([1.0, 1.0], [0.0, 1.0]) - 1.0) < 1e-9
    # straight line y=x over x in [0,1] -> area 0.5
    assert abs(trapezoidal_area([0.0, 1.0], [0.0, 1.0]) - 0.5) < 1e-9
    # matches numpy's own trapezoid rule on a non-trivial curve, whichever name this numpy version exposes
    y, x = [0.1, 0.4, 0.3, 0.6], [0.01, 0.1, 0.5, 1.0]
    np_trapz = getattr(np, "trapezoid", None) or np.trapz
    assert abs(trapezoidal_area(y, x) - float(np_trapz(y, x))) < 1e-9


def test_fractions_match_experimental_spec_section_22():
    assert FRACTIONS == [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def test_nested_fraction_indices_are_strictly_nested():
    n = 500
    idx = nested_fraction_indices(n, FRACTIONS, seed=0)
    sets = {f: set(idx[f].tolist()) for f in FRACTIONS}
    for smaller, bigger in zip(FRACTIONS, FRACTIONS[1:]):
        assert sets[smaller] <= sets[bigger], f"{smaller} subset not contained in {bigger} subset"
    assert len(sets[1.00]) == n


def test_nested_fraction_indices_sizes_match_requested_fractions():
    n = 522   # matches the real dataset's train split size
    idx = nested_fraction_indices(n, FRACTIONS, seed=0)
    for frac in FRACTIONS:
        expected = max(1, round(n * frac))
        assert len(idx[frac]) == expected


def test_smallest_fraction_never_empty_even_for_tiny_n():
    idx = nested_fraction_indices(3, [0.01, 1.0], seed=0)
    assert len(idx[0.01]) >= 1


def test_different_seeds_give_different_orderings_but_same_nesting_property():
    n = 200
    idx_a = nested_fraction_indices(n, FRACTIONS, seed=0)
    idx_b = nested_fraction_indices(n, FRACTIONS, seed=1)
    assert not np.array_equal(idx_a[0.01], idx_b[0.01])
    for smaller, bigger in zip(FRACTIONS, FRACTIONS[1:]):
        assert set(idx_b[smaller].tolist()) <= set(idx_b[bigger].tolist())
