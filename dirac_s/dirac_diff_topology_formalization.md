# DIRAC-S — Differentiable Topology Follow-On: Formalization

**Status: tested end-to-end, 2026-08-21. Result: real but small, inconclusive at n=3 — not recommended for
further compute without a specific reason to expect it'll resolve.** This was a new experiment on top of the
already-complete, signed-off DIRAC-S result (`gate_reports/gate_6.json`, `human_signoff: GO`, 2026-08-17). It
never modified `dirac_formalization.md` (frozen) or re-opened Gate 6 — it named a follow-on hypothesis Gate
5/6 left open (does a materially stronger, differentiable topology signal beat the existing indirect
`PHPredictHead` regression?) and ran it through this vault's full discipline: Section 5's four CPU-only Gate-0
sanity checks (`tests/test_topology_diff.py`, 5/5 pass), a real-data smoke gate at two `LAM_TD` values
(`gate_reports/gate_diff_topology_smoke.json` at `LAM_TD=1.0` — PASSes on `loss_total` alone but
`loss_topology_diff` itself *rises* 0.1997→0.3859, a red flag; `gate_diff_topology_smoke_lamtd0p01.json` at
the rescaled weight — genuinely converges, 0.462→0.364), a mechanism diagnostic
(`jobs/diagnose_pairing_stability.py` — confirmed gudhi's persistence pairing churns almost every SGD step,
0/59 consecutive steps identical, explaining the high floor without fully blocking progress), and finally the
real comparison this whole thread was built to answer: `gate_reports/gate_diff_topology_h2h3c_comparison.json`
reran the H2/H3-C probe methodology (3 probe seeds, same extraction/target as `gate_5_h3c_topology.json` /
`gate_5_h2_data_efficiency.json`) against the `LAM_TD=0.01` checkpoint vs. D_full's own recorded numbers.

**Result: H3-C macro-Spearman +0.014 (0.7257 vs. 0.7113), H2 AULC +0.009 (0.7145 vs. 0.7053), same direction
on all 3 seeds for both metrics — but both deltas sit inside the between-seed noise band** (H3-C per-seed
range ~0.025-0.033; H2 per-seed range ~0.020-0.029). 3/3-same-direction on two related metrics is a real,
non-random pattern (not nothing), but it is not a statistically resolved effect at n=3, and this vault's own
convention throughout Gate 5/6 uses exactly 3 probe seeds as its standard, not more — this thread doesn't get
a different standard just because its result is ambiguous. **Recommendation, not a unilateral decision (human
call per this vault's own gating discipline)**: do not spend further GPU chasing this specific effect size
without a concrete reason to expect it resolves into something larger (e.g. the diagram-matching alternative
in Section 3 below, which changes the mechanism rather than just re-sampling noise). The differentiable
topology loss does not change DIRAC-S's central, already-published finding (Gate 5/6: the advantage is a
training-signal effect carried by `L_topology`, not evidence for the structured-latent hypothesis generally)
— it is at best a minor, unresolved positive footnote to that finding, not a revision of it.

---

## 1. What Gate 5/6 actually found, and the gap this leaves open

`gate_reports/gate_5.json` / `gate_6.json` (summarized in `CLAUDE.md`): DIRAC-S's entire measured advantage
over parameter-matched CNN/ViT baselines is carried almost entirely by **`L_topology`** —
`D_-topology` (remove only that loss term) collapses both H2 (data efficiency) and H3-C (topology probing)
to baseline; `D_-complex` and `D_-prob` (psi's phase, rho's mixture construction) change essentially
nothing; `D_-graph` shows a real but partial drop. The sharpened, signed-off claim is: **DIRAC-S's edge is a
training-signal effect (explicit topology-prediction supervision), not evidence for the structured-latent
architecture hypothesis in general.**

That claim was never in question about *whether* topology supervision matters — it clearly does. What it
leaves open is *how good the topology supervision itself is*. As currently implemented
(`model/topology.py::compute_persistence_summary`, `model/losses.py::PHPredictHead`/`l_topology`), it is:

1. **A frozen, non-differentiable feature extractor.** `gudhi.CubicalComplex` runs outside autograd
   (`model/losses.py::l_topology` docstring: "never backprop through gudhi itself"). Gradient never touches
   the actual persistence computation.
2. **An indirect target.** `PHPredictHead` predicts a 9-number `[count, sum, max]`-per-`H0/H1/H2`,
   `log1p`-transformed summary of the real input's topology, from pooled `(psi, rho)` — not from anything the
   model actually *outputs*. The model is taught to *know about* the input's topology, never to *reproduce*
   it in its own reconstruction.

Given that this one loss term is doing essentially all of DIRAC-S's work, the natural next question is
whether a **materially stronger topological training signal** — one that is differentiable and acts directly
on what the model produces — buys anything beyond the current indirect-regression version. That is what this
document formalizes.

---

## 2. Proposed construction: a differentiable persistence-summary loss on the decoder's own reconstruction

`jobs/pretrain_dirac_job.py` already computes `recon` — `Decoder3D`'s full-resolution reconstruction of the
input volume, used today for `L_recon`/`L_masked` (`model/losses.py::Decoder3D`, `ZBroadcast`). This
construction adds a second signal computed from that same `recon` tensor: **the reconstruction's own
persistence summary, computed differentiably, regressed toward the same frozen target `ph_vector` that
`PHPredictHead` already regresses toward.**

### 2a. The differentiability mechanism — live-verified, not assumed

The obstacle to differentiating through `gudhi.CubicalComplex` is not that persistence is fundamentally
non-differentiable — it is a **piecewise-constant pairing** (which cell pairs with which) **composed with a
selection of input values** (the birth/death filtration values *are* specific input entries). The pairing
structure is locally constant almost everywhere (standard result, e.g. Gabrielsson & Nelson, *A Topology
Layer for Machine Learning*, 2020); only the values need gradient, and those are recovered by construction.

Concretely, verified directly against this project's own `gudhi` install (3.13.0) before writing this
section, not assumed from the docs: for a `CubicalComplex` built the way `compute_persistence_summary`
already builds it (`top_dimensional_cells=(-volume).flatten()`, no explicit lower-dimensional cell
construction), `cc.cofaces_of_persistence_pairs()` returns, **for every homology dimension including H1 and
H2** (checked on a 10×10×10 synthetic volume, all 151 H1 pairs and all 81 H2 pairs), a `(birth_idx,
death_idx)` pair whose values **exactly** index into the *flattened original voxel array* — i.e.
`flat[birth_idx]` and `flat[death_idx]` reproduce `persistence_intervals_in_dimension(dim)` bit-for-bit. This
is gudhi's own "coface" convention (the API name says exactly this: it reports each critical cell's achieving
top-dimensional coface, not an internal complex-cell index), and it means **no custom autograd.Function and
no reimplementation of persistence is needed** — plain PyTorch fancy-indexing (`tensor[idx]`) is already a
differentiable gather op, so:

```
pairing = gudhi_pairing(recon_small.detach().cpu().numpy())     # no_grad: gudhi call, returns (birth_idx, death_idx) per dim
flat = recon_small.flatten()                                     # differentiable — recon_small IS the decoder output
lifetimes = flat[death_idx] - flat[birth_idx]                    # differentiable gather+subtract, real gradient
```

reproduces the exact persistence lifetimes gudhi itself reports, with gradient flowing back through
`recon_small` into the decoder and, via `ZBroadcast`, into `z`/`rho`/`graph`/the encoder — the direct path
`L_topology`'s indirect regression never had.

### 2b. Matching the existing target's resolution

`materialize_ph_vectors_local`/`_compute_one_ph` (`data/local_cache.py`) computes the frozen target via
`scipy.ndimage.zoom(vol, downsample/vol.shape[0], order=1)` at `TOPOLOGY_DOWNSAMPLE=48`
(`jobs/pretrain_dirac_job.py`). `recon` is full-resolution. For an apples-to-apples comparison, `recon` must
be downsampled the same way before its differentiable summary is computed. `order=1` zoom is trilinear
interpolation — `F.interpolate(recon, size=(48,48,48), mode="trilinear", align_corners=False)` is the
differentiable PyTorch equivalent, not an approximation invented for this document. This is a real,
checkable design point (named per this vault's convention, not assumed harmless) — Gate 0 for this
experiment (Section 5 below) explicitly checks the two downsampling paths don't diverge enough to matter.

### 2c. The statistics: count is not differentiable, sum and max are

`compute_persistence_summary` reports `[count, sum, max]` of finite lifetimes per dimension. For a *fixed*
pairing (frozen for one forward pass, per Section 2a):
- **`count`** is the cardinality of the pairing set — a discrete, piecewise-constant integer. Treated as a
  constant this forward pass (standard convention in the differentiable-persistence literature — only the
  continuous statistics of an already-fixed pairing are optimized, never the combinatorial pairing itself).
  No gradient flows through it; it still contributes to the MSE loss value the same as today.
- **`sum`** is a plain differentiable sum over `lifetimes`.
- **`max`** is `torch.max` over `lifetimes` — standard subgradient, gradient flows only to the argmax
  lifetime's two indexing positions.

Both `sum` and `max` go through `log1p`, matching `compute_persistence_summary` exactly (§ topology.py:
"non-negative by construction... log1p is safe everywhere"); `lifetimes >= 0` holds here too, for the same
reason (each pairing is recomputed fresh from the *current* `recon_small` every forward pass, so it is
always a valid pairing for the volume it was just extracted from — the pairing goes stale only *across* — not
within — a step, which is exactly the "locally constant a.e." property Section 2a relies on).

### 2d. The loss

```
recon_small = F.interpolate(recon, size=(D,D,D), mode="trilinear", align_corners=False)   # D = TOPOLOGY_DOWNSAMPLE
pred_summary = differentiable_persistence_summary(recon_small, max_dim=TOPOLOGY_MAX_DIM)   # model/topology_diff.py
loss_topology_diff = l_topology(pred_summary, ph_vector)   # reuse the existing MSE loss verbatim — same function, same target
```

Added to `L_DIRAC` as a fifth term, **alongside**, not replacing, the existing `l_topology(PHPredictHead(...),
ph_vector)` — this experiment tests whether a direct signal adds anything *on top of* the indirect one Gate
6 already validated, not whether it can replace it (replacing it is a strictly larger, riskier change with no
current evidence behind it).

---

## 3. Rejected / deferred alternatives — named explicitly, not silently dropped

- **Full persistence-diagram matching (Wasserstein/bottleneck distance).** The theoretically "correct"
  differentiable-topology loss compares entire diagrams via optimal matching (as in `TopologyLayer`,
  topological autoencoders). Rejected for this first test: it needs a matching algorithm (with its own
  diagonal-point conventions and failure modes), is more expensive, and — critically — changes two things at
  once (matching-based distance *and* differentiability) versus the current summary-vector regression, which
  makes a clean ablation against Gate 6's existing `L_topology` impossible. Section 2's construction changes
  exactly one thing (differentiability) and keeps the exact same 9-number representation and MSE form
  `L_topology` already uses — deferred here as a candidate second-generation experiment if Section 2's
  simpler version shows real signal.
- **A custom `torch.autograd.Function` for the persistence layer.** Unnecessary — Section 2a's live
  verification shows plain PyTorch indexing already differentiates correctly through the gather; adding a
  hand-written backward would be unvalidated extra machinery solving a problem that doesn't exist.
- **Applying this directly to `psi`/`rho` instead of the decoder's reconstruction.** More directly tests
  DIRAC's "structured latent" hypothesis (does the latent field *itself* have to be topologically faithful,
  not just the eventual reconstruction) but needs a real design decision this document is not making blind:
  what scalar field derived from a complex `psi`/mixture `rho` per token even has a meaningful voxel-grid
  topology (`|psi|^2`? the dominant mixture component's projection? entropy `S(rho)` as a scalar field?).
  Named and deferred, not attempted here — Section 2's decoder-output version reuses an already-existing,
  already-validated tensor (`recon`) and needs no new design decision to be well-posed.
- **Backpropagating through `gudhi` itself (e.g. a hypothetical autograd-aware persistence library).** Not
  needed per Section 2a, and no such library is a current project dependency — introducing one would violate
  this vault's minimal-dependency convention for a benefit Section 2a's plain-indexing approach already gets
  for free.

---

## 4. What this deliberately does not add

- No change to `D_-topology`/`D_-complex`/`D_-prob`/`D_-graph`/`D_classical` or their signed-off gate
  reports — those results stand as reported.
- No modification to `model/dirac_model.py`, `model/losses.py`, `model/topology.py`, or any other frozen
  Gate-6 file. This experiment's code lives entirely in new files (`model/topology_diff.py`, a new job/test),
  matching how the five existing ablations each got their own `train_ablation_*_job.py` rather than mutating
  the base model in place.
- No claim, yet, that this beats `D_full`. Section 5's gates are local, synthetic, CPU-only sanity checks —
  correctness of the mechanism, not evidence it helps. A real comparison needs its own H2/H3-C-style probe
  run against a real retrain, which is real compute and requires its own explicit human GO, same as every
  other GPU-touching step in this vault.

---

## 5. Falsification / sanity gates (all CPU-only, synthetic or already-downloaded data — no new GPU spend)

Run before any retrain is even proposed:

1. **Consistency.** `differentiable_persistence_summary(x)`'s value must exactly match
   `compute_persistence_summary(x.numpy())`'s value (both derived from the same pairing, Section 2a) on
   several random synthetic volumes, across `max_dim=0,1,2`. *Fail → the gather/indexing logic is wrong,
   stop before anything else.*
2. **Gradient sanity.** Finite-difference check on a small (e.g. 8³) synthetic volume: perturbing a single
   voxel that participates in a real pairing must change `differentiable_persistence_summary(x).sum()` by
   the amount the analytic gradient predicts, within numerical tolerance. *Fail → gradient path is not what
   Section 2a claims, stop.*
3. **End-to-end wiring.** On synthetic data through a real `DiracModel` + `Decoder3D` + `ZBroadcast`
   (mirroring `tests/test_ablation_topology.py`'s pattern): `loss_topology_diff.backward()` must produce real,
   nonzero gradient in the decoder, `ZBroadcast`, `fusion`, `graph_head`, and the encoder — the whole point of
   Section 2's construction is that this path reaches further back than `PHPredictHead`'s regression alone.
4. **Resolution-mismatch check (Section 2b).** On a real (already-materialized, no new download) case's full
   volume: compare `compute_persistence_summary(ndimage.zoom(vol, ..., order=1))` against
   `differentiable_persistence_summary(F.interpolate(vol, ..., mode="trilinear"))` on the *same* real volume
   (not a reconstruction — isolating the resize-method question from the reconstruction-quality question).
   Large, systematic divergence here would mean Section 2b's resize substitution needs revisiting before
   trusting any training signal built on it.

Only after all four pass, with a human reviewing the actual numbers (not just PASS/FAIL), does adding
`loss_topology_diff` to a real `pretrain_dirac_job.py`-style smoke run become the next, separately-gated
step — matching this vault's Gate-0-before-Gate-1 discipline everywhere else.
