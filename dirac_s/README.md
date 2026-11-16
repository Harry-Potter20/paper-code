---
license: mit
tags:
  - medical-imaging
  - self-supervised-learning
  - 3d-ct
  - structured-representation
  - ablation-study
  - negative-results
library_name: pytorch
---

# DIRAC-S — Quantum-Inspired Structured Representation for Medical CT

Code, gated evaluation, and 3-seed reproducibility check for a research-scale (1.55M param) 3D CT
foundation model testing whether an explicitly structured latent state — complex representation +
probabilistic (density-matrix-style) state + relational graph + topology — beats parameter-matched CNN/ViT
baselines at matched capacity.

> **Reframed contribution (see `gate_reports/SUMMARY.md`).** DIRAC-S does beat CNN/ViT, reproducibly across
> 3 backbone seeds, on data efficiency (H2) and topology recovery (H3-C). But the load-bearing result is
> what the 5-ablation disentangling (`gate_reports/gate_5.json`) and the seed-reproducibility check
> (`gate_reports/gate_6.json`) found underneath that: the advantage is a **training-signal effect**
> (`L_topology`, one explicit auxiliary objective), not a structural one. Removing only `L_topology` —
> leaving the complex/probabilistic/graph machinery fully intact — collapses both metrics to baseline
> level, reproducibly. The central "structured representation provides architecture-level advantage"
> hypothesis is refuted at this scale. See `gate_reports/SUMMARY.md` for the full breakdown, including what
> genuinely does and doesn't hold up (H1/H3-A/H3-B null, H4 an active FAIL, H5 a real DIRAC-specific
> weakness).

## Headline numbers

| | |
|---|---|
| Data efficiency (H2 AULC, 3-seed) | DIRAC **0.704±0.006** vs. CNN 0.533±0.006 vs. ViT 0.531±0.020 |
| Topology recovery (H3-C, 3-seed) | DIRAC **0.716±0.008** vs. CNN 0.573±0.007 vs. ViT 0.566±0.020 |
| `D_-topology` ablation (same metrics) | 0.530±0.029 / 0.549±0.037 — **collapses to baseline level** |
| Parameter matching | DIRAC 1,552,520 vs. CNN/ViT, both within 5% |
| Uncertainty check (H4) | FAIL — Spearman −0.669, backwards; reproduces a known sheaf-project failure mode |
| Robustness (H5) | No general advantage; DIRAC uniquely least robust to additive noise at high severity |

## Layout

```
dirac_formalization.md              frozen math (ψ/ρ/G/T/z), do not edit in place
DIRAC-S_Experimental_Specification_v1.0.md   frozen pre-registration-style protocol, do not edit in place
IMPLEMENTATION_SPEC.md              gated build spec for a coding agent
model/            ComplexEncoder, DensityHead, GraphHead, TopologyHead, FusionHead, DiracModel
baselines/        CNN3DBaseline, ViT3DBaseline (independent architectures, own recipes)
data/             NIfTI load/preprocess, manifest + split hash, leak guard (ported from covtoken)
jobs/             all experiments as Hugging Face Jobs (materialize, pretrain, baselines, ablations, probes)
gate_reports/     machine-readable per-gate decision records + SUMMARY.md (this file's companion)
tests/            invariant checks (ψ/ρ PSD+trace1, graph connectivity, param count, label leak)
configs/          param_ledger.json, thresholds.lock.json, dataset_manifest.lock.json
```

`Med_Architecture_Research/dirac_m/` (sibling directory) is the explicitly gated next step — a scale-up
study testing whether the training-signal-not-structure finding above holds, weakens, or reverses with
~1M training examples across multiple anatomical regions instead of 801 cases of single-region oncology CT.
Does not start until this project's Gate 6 (above) reaches human GO, per `IMPLEMENTATION_SPEC.md`'s own
phase-gating rule.

## Reproducibility

All experiments ran as Hugging Face Jobs (t4-medium). Datasets: KiTS23 (389 cases), MSD Task03 Liver (131),
MSD Task07 Pancreas (281) — 801 total, 70/15/15 patient-level split via `sha256(case_id) % 100` (reused from
`covtoken/data/leak_guard.py`). `tests/test_label_leak.py` fails the build if a label reaches subspace
construction. Code mounted read-only from `hf://datasets/Chucks90/dirac-code`; data/checkpoints from
`hf://buckets/Chucks90/jobs-artifacts`.

Decision records: `gate_reports/gate_0.json` through `gate_6.json`, `gate_reports/SUMMARY.md`. Locked
thresholds: `configs/thresholds.lock.json`. Locked parameter ledger: `configs/param_ledger.json`.
