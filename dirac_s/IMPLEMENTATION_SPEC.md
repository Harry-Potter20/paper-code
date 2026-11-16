# DIRAC-S Implementation Spec (gated build)

Target reader: an autonomous coding agent. The agent implements one phase, runs the gate, emits a
machine-readable gate report, and HALTS for human go/no-go. This spec operationalizes
`dirac_formalization.md` (the frozen math) and `DIRAC-S_Experimental_Specification_v1.0.md` (the frozen
experimental protocol) into buildable, gated phases. Read both before writing any code.

---

## 0. Operating protocol for the agent

1. Implement phases strictly in order (0b → 1 → 2 → 3 → 4 → 5 → 6). Do not start a phase before the prior
   gate's `human_signoff` is `"GO"`. Phase 0 (the two frozen documents) is already complete as of this spec's
   creation — Phase 0b is the first phase requiring code.
2. Every phase ends by writing `gate_reports/gate_<N>.json` (schema: §8) and printing a clear HALT message.
   A human reviews the report and issues an explicit GO before the next phase starts.
3. Every gate has exactly three possible outcomes: `PASS`, `FALLBACK` (a named, pre-specified degraded path
   exists and is documented in the gate table), or `FAIL` (write up why, per Definition of Done §9 — a FAIL is
   a valid, reportable scientific outcome, not a bug to hide).
4. No threshold is "roughly met." A gate either passes its stated comparator or it doesn't.
5. Thresholds tagged `[CALIBRATE]` are placeholders, replaced by data-driven values computed in Phase 1
   (dataset statistics) or Phase 2 (real-data unit-test statistics) before they are used as a gate, then locked
   to `configs/thresholds.lock.json` and immutable thereafter. Thresholds tagged `[FIXED]` are decision
   constants set by the human and never changed by the agent.
6. **"The contribution is the structure, not the parameter count" is a hard invariant, not a target.** Every
   result comparing DIRAC to a baseline (Gates 4-6) must report the exact parameter counts of both models per
   `dirac_formalization.md` §9's counting convention. Any code path that reports an H1/H2/H3 result without an
   accompanying parameter-count table is a spec violation and must fail CI — this is DIRAC's equivalent of
   covtoken's label-leak invariant: a load-bearing guarantee that must be enforced structurally, not by
   discipline alone.
7. `ρ` must never be reported as validated (PSD/trace-1) without the actual eigenvalue/trace check run and
   logged — no eyeballing shapes.

---

## 1. Success ledger

| Gate | Question | Metric | Comparator | Threshold | Test | Decision | If FAIL |
|---|---|---|---|---|---|---|---|
| 0 | Do ψ/ρ/G/T satisfy their construction invariants on synthetic input? | PSD min-eigenvalue, trace error, graph orphan count, PH finite-ness, param-count delta | vs. exact constants | eig ≥ −1e-6 [FIXED]; \|trace−1\| ≤ 1e-5 [FIXED]; 0 orphans [FIXED]; 0 NaN/Inf [FIXED]; param delta ≤ 5% [CALIBRATE] | `tests/test_rho_psd_trace1.py`, `test_graph_connectivity.py`, `test_param_count.py` | PASS iff all five hold | Fix the failing construction; no FALLBACK — these are correctness bugs, not judgment calls |
| 1 | Is the dataset trio usable and leak-free? | QC pass rate, orientation/spacing normalization rate, split-leakage count | vs. exact constants | QC pass ≥ 90% [CALIBRATE→locked]; leakage = 0 cases [FIXED] | `tests/test_label_leak.py` (ported), manifest QC report | PASS iff leakage=0 AND QC≥threshold | FALLBACK: drop the failing dataset from the trio, proceed with 2; FAIL only if <2 datasets pass |
| 2 | Do ψ/ρ/G/T hold on real preprocessed volumes (not just synthetic)? | Same five checks as Gate 0, on real data | vs. Gate 0's constants | same as Gate 0 | Same test files, real-data fixture | PASS iff all five hold on ≥95% [CALIBRATE] of a real calibration subset | FALLBACK: exclude the specific failure mode's case type (e.g. degenerate/empty-region cases) from training with a documented `data_gaps` entry; FAIL if failure rate >20% |
| 3 | Does pretraining converge without instability? | Loss curve (all 4 terms), gradient norm, NaN count | vs. smoke-scale run | loss decreasing over final 20% of smoke steps [CALIBRATE]; 0 NaN/Inf steps [FIXED] | `jobs/pretrain_dirac_job.py` smoke mode (data subset, few epochs) before the full run | PASS iff smoke run is stable | FALLBACK: reduce a named loss weight or lower LR per a pre-specified sweep; FAIL if instability persists across the sweep |
| 4 | Are baselines genuinely parameter-matched? | \|N_DIRAC − N_CNN\|, \|N_DIRAC − N_ViT\| (exact counts, §0 rule 6 convention) | vs. locked tolerance | ≤5% [CALIBRATE→locked in Phase 0b] | `model/param_count.py` ledger, `configs/param_ledger.json` | PASS iff both within tolerance | FALLBACK: retune baseline width (not depth) within the calibrated sweep; FAIL only if no width choice lands in tolerance |
| 5 | Do H1-H5 hold, and is H4's uncertainty check clean? | Per-hypothesis: probing AUROC/accuracy deltas (H1), data-efficiency AUC (H2), structural-task deltas (H3), uncertainty-error AUROC + magnitude-conflation partial-correlation (H4), degradation deltas under perturbation (H5) | vs. matched CNN/ViT | `[CALIBRATE]`, locked after Phase 1 dataset stats are available | `jobs/probe_linear_job.py`, `probe_nonlinear_job.py`, `uncertainty_gate_job.py`, robustness jobs (TBD Phase 5) | Per-hypothesis PASS/FAIL, reported individually — no single combined verdict (§0 rule 6, §9) | H0 (no advantage) is a legitimate, reportable outcome, not a FALLBACK path — see Definition of Done |
| 6 | Do results replicate across seeds? | Mean, std, 95% CI across seeds | n≥3 [FIXED] | CI excludes the matched-baseline point estimate, where a PASS is claimed | Paired bootstrap or equivalent, `configs/thresholds.lock.json`-recorded seeds | PASS iff the Gate 5 verdict is stable across seeds | Report the instability explicitly; do not average away a seed-dependent result |

---

## 2. Repository layout

```
Med_Architecture_Research/dirac/
  DIRAC-S_Experimental_Specification_v1.0.md   # frozen, do not edit
  dirac_formalization.md                        # frozen, do not edit (Phase 0)
  IMPLEMENTATION_SPEC.md                        # this file, frozen once Phase 0 signs off
  research_specs/                               # APPEND-ONLY: if the frozen docs need to evolve, add a new
                                                  # versioned file here — never edit the originals in place
  model/
    complex_encoder.py       # PHASE 0b. psi(X): conv stem + transformer; ComplexState(re, im)
    density_state.py         # PHASE 0b. rho(X): mixture-of-K construction; von Neumann entropy
    graph.py                  # PHASE 0b. G: patch-adjacency + top-k edges; 2-layer GCN
    graph_anatomical.py       # PHASE 5+ ABLATION ONLY. G Construction B — do not wire into v1 forward pass
    topology.py                # PHASE 0b. T: gudhi.CubicalComplex 3D wrapper; PH summary vector
    losses.py                   # PHASE 0b (shapes only) / PHASE 3 (real): L_recon/masked/struct/topology
    fusion.py                    # PHASE 0b. z = f(rho, G, T)
    dirac_model.py                # PHASE 0b. top-level nn.Module; D_classical ablation flag lives here
    param_count.py                 # PHASE 0b. Section 9 exact parameter ledger
  baselines/
    cnn3d.py                        # PHASE 4. independent architecture, own recipe
    vit3d.py                         # PHASE 4. independent architecture, own recipe
  data/
    volume_io.py                      # PHASE 1. NIfTI load, reorientation, resampling
    preprocess.py                      # PHASE 1. HU handling, cropping, fixed-grid tensor formatting
    manifest.py                         # PHASE 1. split_of() reused from covtoken; manifest schema
    leak_guard.py                        # PHASE 1. ported near-verbatim from covtoken/data/leak_guard.py
  jobs/
    materialize_kits_job.py               # PHASE 1
    materialize_msd_liver_job.py           # PHASE 1
    materialize_msd_pancreas_job.py         # PHASE 1
    pretrain_dirac_job.py                    # PHASE 3
    train_baseline_cnn_job.py                 # PHASE 4
    train_baseline_vit_job.py                  # PHASE 4
    probe_linear_job.py                         # PHASE 5
    probe_nonlinear_job.py                       # PHASE 5
    uncertainty_gate_job.py                       # PHASE 5. H4 gate + magnitude-conflation check
  gate_reports/
    gate_0.json ... gate_6.json                    # schema: §8. Narrative postmortems (gate_N_notes.md)
                                                     # welcome alongside the JSON for any FALLBACK/FAIL.
  tests/
    test_rho_psd_trace1.py                          # PHASE 0b
    test_param_count.py                              # PHASE 0b
    test_graph_connectivity.py                        # PHASE 0b
    test_label_leak.py                                 # PHASE 1. ported from covtoken
  configs/
    dataset_manifest.lock.json                          # PHASE 1
    param_ledger.json                                    # PHASE 0b
    thresholds.lock.json                                  # locked [CALIBRATE] values, append-only
  README.md                                                # written once Gate 0 passes
```

---

## 3. Phase plan (gated)

**Phase 0 — COMPLETE.** `dirac_formalization.md` + this file written and signed off. No code.

**Phase 0b — Parameter accounting + scaffolding.** Implement `model/*.py` (forward pass on synthetic 3D
tensors — no real dataset dependency), `baselines/*.py` skeletons, `param_count.py`. Implement
`tests/test_rho_psd_trace1.py`, `test_graph_connectivity.py`, `test_param_count.py`. Run Gate 0. Lock the
parameter-matching tolerance and the initial K, N, d constants into `configs/param_ledger.json`. HALT.

**Phase 1 — Dataset select+freeze, preprocessing.** Implement `data/volume_io.py`, `preprocess.py`,
`manifest.py` (reusing covtoken's split-hash function verbatim), `leak_guard.py` (ported), and the three
`materialize_*_job.py` files (adapted from covtoken's — keep the download/resume scaffolding and case-list
logic, replace the per-slice PNG-save loop with a call into `volume_io`+`preprocess` that saves one 3D tensor
per case). Run Gate 1. Freeze `configs/dataset_manifest.lock.json`. HALT.

**Phase 2 — Unit tests for ρ/G/T on real data.** Re-run Gate 0's exact checks against real preprocessed
volumes from the frozen manifest, not synthetic tensors. This phase exists specifically to catch real-data
edge cases (degenerate/empty anatomical regions, HU outliers, near-empty foreground masks) before committing
to an expensive pretraining run — deliberately ordered before Phase 3 pretraining, a documented deviation from
the experimental spec's literal step order (§34 lists pretraining before these unit tests; this spec inverts
them, mirroring covtoken's own gate-before-expensive-training discipline). Run Gate 2. HALT.

**Phase 3 — Pretraining.** Implement `losses.py`'s real forms (Gate 0b only checked shapes) and
`pretrain_dirac_job.py`. Run a smoke-scale pass (data subset, few epochs) first as Gate 3's PASS condition
before committing to the full run. Full run does not require a separate gate — its outputs feed Phase 4/5's
gates directly. HALT after the smoke gate; human GO authorizes the full run's compute spend explicitly (this is
the phase with real GPU-hour cost, and the spec's own H0-honesty stance means that spend should be authorized
knowingly, not on autopilot).

**Phase 4 — Baselines.** Implement `train_baseline_cnn_job.py`, `train_baseline_vit_job.py`. Confirm parameter
matching against the Phase 0b ledger with real trained models (not just architecture skeletons). Run Gate 4.
HALT.

**Phase 5 — Probing / transfer / uncertainty / robustness / ablations.** Implement `probe_linear_job.py`,
`probe_nonlinear_job.py`, `uncertainty_gate_job.py` (the `dirac_formalization.md` §8 magnitude-conflation check
— this is where it actually executes, using real prediction-error data). Robustness/OOD and ablation job files
are named but not fully specified here — follow experimental-spec §19/§20 directly when this phase starts;
do not invent additional ablations beyond `D_-complex`, `D_-prob`, `D_-graph`, `D_-topology`, `D_classical`
without an explicit, separately-justified reason. Run Gate 5 — report H1-H5 individually, never as one combined
score. HALT.

**Phase 6 — Multi-seed, freeze, paper decision.** Re-run the critical Gate-5 experiments across ≥3 seeds. Run
Gate 6. Write `gate_reports/SUMMARY.md` stating plainly which hypotheses were supported, which were not, and
whether the evidence meets the experimental spec's §27 publication-readiness bar. If a headline claim is
falsified, document it in the README the way covtoken's own README documents its falsified core hypothesis —
do not quietly bury a negative result. HALT — a human makes the paper go/no-go call.

**Phase 7 — DIRAC-M scale-up. NOT PART OF THIS SPEC.** Out of scope; requires its own spec, evidence-gated per
experimental-spec §30, only after a human GO on Phase 6.

---

## 4. Key interfaces (stable contracts)

```python
from dataclasses import dataclass
from typing import Protocol
import torch

@dataclass
class ComplexState:
    """psi(X), per dirac_formalization.md Section 2 — kept as a (re, im) real-tensor pair; only
    materialized as torch.complex at sites that need genuine complex arithmetic."""
    re: torch.Tensor   # (B, N, d)
    im: torch.Tensor   # (B, N, d)

@dataclass
class DensityState:
    """rho(X), per Section 3 Construction B (mixture-of-K). Consumers must treat `weights` and
    `basis` as the source of truth — never reconstruct a materialized (N, d, d) rho unless a specific
    computation genuinely requires it (Section 3's efficiency note: entropy is a K x K eigendecomposition,
    not a d x d one)."""
    weights: torch.Tensor   # (B, N, K) — pi_{i,k}, softmax-normalized
    basis: ComplexState      # phi_{i,k}, K stacked complex projections per token

    def entropy(self) -> torch.Tensor:
        """Von Neumann entropy S(rho_i), per token. Section 3 + Section 8 (uncertainty)."""
        ...

    def check_psd_trace1(self, eig_tol: float = -1e-6, trace_tol: float = 1e-5) -> bool:
        """Gate 0/2's correctness check — must be called and logged, never assumed."""
        ...

@dataclass
class RelationalGraph:
    """G, per Section 4 Construction A. Nodes are psi's own N tokens."""
    adjacency: torch.Tensor   # (B, N, N), lattice + top-k feature-similarity edges
    node_features: torch.Tensor   # (B, N, d), post-GCN

@dataclass
class TopologySummary:
    """T(X), per Section 5. Computed from raw X, not from rho — see Section 5's chicken-and-egg note."""
    ph_vector: torch.Tensor   # (B, m), fixed-length persistence summary

class DiracModel(Protocol):
    """Top-level module. `classical_ablation=True` strips complex/prob/graph/topology branches for the
    D_classical ablation (Section 9) — this flag path must NOT be reused for the parameter-matched CNN/ViT
    baselines, which are separate architectures in baselines/."""
    def forward(self, x: torch.Tensor, classical_ablation: bool = False) -> dict:
        """Returns {"psi": ComplexState, "rho": DensityState, "graph": RelationalGraph,
        "topology": TopologySummary, "z": torch.Tensor}."""
        ...
```

---

## 5. Anti-goals (do not do)

- Do not backpropagate through `gudhi`. `L_topology` is a regression to a stop-gradient target (formalization
  §5, §7) — no differentiable-PH layer is in scope for DIRAC-S.
- Do not implement Construction B graphs (`graph_anatomical.py`) as part of the v1 forward pass — ablation-only,
  Phase 5+, per formalization §4.
- Do not implement Construction C's full-rank Cholesky ρ — deferred to DIRAC-M, per formalization §3.
- Do not add `torch_geometric`/`dgl` as a dependency. The from-scratch 2-layer GCN in `graph.py` is sufficient
  for v1's dense N=512 adjacency.
- Do not report any H1-H5 result without the accompanying exact parameter-count table (§0 rule 6).
- Do not treat H0 as a bug. A gate-5 result showing no advantage over matched baselines is written up per §9,
  not retried with loosened thresholds until it passes.
- Do not skip the magnitude-conflation check (formalization §8) when reporting any uncertainty result — this
  is the one check in this spec with a directly documented prior failure (`gate_reports/gate_2.json`) behind
  it; skipping it would repeat a known mistake, not just an unverified one.
- Do not start Phase 3's full pretraining run before Phase 2's real-data gate passes — an expensive run around
  an unvalidated ρ/G/T is exactly the failure mode Gate 2 exists to prevent cheaply.
- Do not edit `DIRAC-S_Experimental_Specification_v1.0.md` or `dirac_formalization.md` in place once Phase 0
  signs off — new versions go in `research_specs/` as append-only revisions.

---

## 6. Statistical methods (single source of truth)

- **Probing (Gate 5, H1):** linear/nonlinear probe accuracy or AUROC, paired bootstrap CI (n≥2000 resamples)
  against the matched baseline's probe on the same split.
- **Data efficiency (Gate 5, H2):** area under the label-fraction learning curve (1/5/10/25/50/100%, per
  experimental spec §22), reported alongside the per-fraction points, not instead of them.
- **Uncertainty (Gate 5, H4):** AUROC/Spearman for error-vs-uncertainty; partial correlation (controlling for
  `‖ψ‖`) for the magnitude-conflation check, per formalization §8 — the same statistic `gate_reports/gate_2.json`
  used (`partial_spearman_D_error_given_entropy`), for direct comparability across the two projects.
- **Robustness (Gate 5, H5):** degradation delta `Δ_DIRAC` vs. `Δ_baseline` under matched perturbation severity,
  per experimental-spec §20-21.
- **Reproducibility (Gate 6):** mean/std/95% CI across seeds (n≥3), exact seeds and software/hardware versions
  logged in every gate report.
- Multiple-comparison correction applied whenever a gate reports more than one hypothesis test simultaneously
  (Gate 5 reports five — H1 through H5 — each requiring its own corrected significance level, not five
  independent unadjusted tests).

---

## 7. Datasets

| Use | Dataset | Notes |
|---|---|---|
| Pretraining + primary eval | KiTS23 (kidney+tumor) | Real working ingestion scaffolding already exists (`covtoken/jobs/materialize_kits_job.py`); reuse download/resume + split-hash, write new 3D-preserving preprocessing |
| Pretraining + primary eval | MSD Task03 Liver | Same reuse pattern via `covtoken`'s MSD scaffolding |
| Pretraining + primary eval | MSD Task07 Pancreas | Same reuse pattern; `covtoken/jobs/materialize_msd_pancreas_job.py` is the closest existing precedent |
| Deferred, not v1 | LIDC-IDRI | `covtoken/data/loaders.py` documents the interim PNG bucket as unreadable with the current token — confirm real pixel access in a dedicated Phase 1 sub-job before including it, don't inherit the gap silently |
| Deferred, not v1 | External distribution-shift test set (spec §21) | Named as a future need; no candidate frozen yet — revisit once the primary trio's Gate 1 passes |

All splits patient-level via the reused `sha256(case_id) % 100 → 70/15/15` hash. `dataset_manifest.lock.json`
frozen at the end of Phase 1; not modified afterward except by adding a new, separately-gated dataset.

---

## 8. Gate report schema

```json
{
  "project": "dirac",
  "gate": 0,
  "phase": "Phase 0b - Parameter accounting + scaffolding",
  "objective": "rho/G/T satisfy their construction invariants on synthetic input; N_DIRAC-S/N_CNN/N_ViT within tolerance",
  "job_id": null,
  "checks": {
    "rho_psd_min_eig": {"value": 0.0, "threshold": -1e-6, "passed": true},
    "rho_trace_error": {"value": 0.0, "threshold": 1e-5, "passed": true},
    "graph_orphan_nodes": {"value": 0, "threshold": 0, "passed": true},
    "topology_finite": {"value": true, "passed": true},
    "param_count_delta_pct": {"dirac_vs_cnn": 0.0, "dirac_vs_vit": 0.0, "threshold": 5.0, "threshold_status": "CALIBRATE->locked", "passed": true}
  },
  "status": "PASS | FALLBACK | FAIL",
  "fallback_path": null,
  "seeds": [0, 1, 2],
  "data_gaps": [],
  "decision_rule": "PASS iff all five checks pass",
  "interpretation": "<prose>",
  "next": "Phase 1: dataset select+freeze, preprocessing",
  "reproduce": "Med_Architecture_Research/dirac/jobs/<job file>.py (HF Job, <flavor>). Module: dirac/model/*.py.",
  "human_signoff": null
}
```

Every field except `human_signoff` is filled by the agent; a phase is complete only when a human sets
`human_signoff` to `"GO"`. Field names for Gate 5's uncertainty check must match `gate_reports/gate_2.json`'s
(`H2a_error_auroc`, `partial_spearman_D_error_given_entropy`-style naming) exactly, for direct cross-project
comparability.

---

## 9. Definition of done

Phase 0b through Phase 4 are done when every listed gate has `status: "PASS"` (or a documented `FALLBACK` that
itself passed) and `human_signoff: "GO"`.

Phase 5 (H1-H5) does **not** require every hypothesis to PASS to be "done" — it requires every hypothesis to be
**honestly reported**, individually, with its own gate-table row and evidence, including H0 as a legitimate
terminal state for any given hypothesis (experimental-spec §26, Failures A-D are explicitly valid scientific
outcomes). A Phase 5 gate report showing "H1 PASS, H2 FAIL, H3 PASS, H4 PASS, H5 FALLBACK" is complete and
correct; it is not blocked on making everything green.

The project as a whole reaches Definition of Done at Phase 6, when: results are reproducible across ≥3 seeds
(Gate 6), every reported comparison carries its exact parameter-count table (§0 rule 6), and
`gate_reports/SUMMARY.md` states plainly, per experimental-spec §27, whether the evidence supports proceeding to
a first paper — regardless of which way that answer goes.
