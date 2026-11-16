---
status: PRE-REGISTERED 2026-11-10, before any gate output is computed. Programme Idea F, step 2.
platform: Kaggle private kernel (PTB-XL, Norwegian athlete DB, PF12RED; ECGFounder MIT).
---

# F step 2: the International-Criteria gate over the foundation model's own named concepts

**Why.** P1's fix was a linear head on hidden features, and it failed: green over-call fell only because normal
over-call rose. Step 2 does not learn a readout. ECGFounder already outputs 150 *named* findings, i.e. interpretable
concept sensors. The International Criteria (Sharma 2017) are a rule over findings, so we implement that rule over the
model's own concepts.

**Concept mapping** (fixed now; full list in `kaggle_f2/f2_kernel.py`):
- **GREEN** (never contributes): sinus bradycardia (incl. marked), sinus arrhythmia, 1st-degree AVB / prolonged AV
  conduction, Mobitz I, incomplete RBBB / rSR′, voltage-only LVH, early repolarisation, ectopic atrial / junctional
  rhythm.
- **BORDERLINE** (abnormal only if ≥ 2): left or right axis deviation (incl. leftward / rightward / abnormal), LAFB,
  LAE, RAE, complete RBBB.
- **RED:** all infarct / injury / acute MI labels, ST depression or elevation, T-wave inversion or nonspecific
  ST/T abnormality, LBBB / IVCD / wide QRS / bifascicular, pre-excitation (WPW, short PR), prolonged QT, PVCs /
  bigeminy, atrial and ventricular tachyarrhythmias, Mobitz II / complete block, non-voltage LVH, RVH, biventricular
  hypertrophy, pacing.
- **Ignored:** normal / borderline / abnormal summary labels, serial-comparison labels, PACs, sinus tachycardia,
  low-voltage QRS, paediatric.

**Gate:** ABNORMAL ⇔ (max RED probability > τ) OR (#BORDERLINE probabilities > τ ≥ 2).
- **τ:** one parameter, set on PTB-XL folds 1–8 to give **90% sensitivity on red ECGs** (P1's red definition: ≥ 1
  finding outside green ∪ {SR, NORM}).
- **Comparator:** ECGFounder's own ABNORMAL output, thresholded at 90% red sensitivity on the same folds.

**Test sets:**
- PTB-XL folds 9–10 (red, green-only, normal);
- Norwegian elite athletes (cardiologist-normal, n = 26);
- PF12RED green-only players (52; player-level bootstrap).

**Pre-registered hypotheses:**
- **F2-H1:** on PTB-XL folds 9–10, the gate's green-only over-call ≤ ½ the comparator's, **and** its normal over-call
  is not higher than the comparator's by > 2 points (P1's failure mode). Red sensitivity reported, target ≈ 0.90.
- **F2-H2:** athlete over-call (Norwegian normal; PF12RED green-only) is reduced ≥ 50% relative to the comparator, in
  both cohorts.
- **Descriptive only:** ancestry breakdown (PF12RED African n = 5; too small for a parity constraint). The pre-planned
  primal-dual parity constraint is deferred until a cohort with adequate ancestry data exists.

## RESULT (Kaggle `f2-criteria-gate`; kernel deleted): FAIL on both hypotheses

| at 90% red sensitivity (thresholds from PTB-XL folds 1–8) | red sens. | green over-call | normal over-call | Norwegian normal | PF12RED green-only |
|---|---|---|---|---|---|
| ECGFounder ABNORMAL output | 0.870 | 0.441 | 0.157 | 0.500 | 0.551 [0.44, 0.66] |
| **Criteria gate (named concepts)** | 0.892 | **0.617** | **0.420** | **0.654** | **0.622** [0.51, 0.73] |

- **F2-H1 FAILS and F2-H2 FAILS.** The gate is *worse* than the raw head on every false-positive measure.
- **Why (diagnostic, not a re-analysis):**
  - Holding 90% red sensitivity with one shared threshold forced τ down to 0.36 across ~100 red concept heads.
  - The max over many weakly firing, uncalibrated concepts (nonspecific ST/T, infarct labels; "lateral infarct" fires
    on athletes) inflates false positives.
  - The named concepts are themselves miscalibrated on athletic and normal ECGs. The over-call lives in the concept
    sensors, not only in the readout, consistent with P1's failed linear head.
- **What this means for P1:** neither a learned linear readout (P1 v2) nor a criteria rule over the model's own named
  findings fixes athlete over-call. The fix must reach the concept / representation level, or use athlete-specific
  calibration data.
- **Possible next step (needs its own prereg):** per-concept calibrated thresholds (each red concept at its own
  specificity on normal controls) instead of one shared τ. Not run here, to avoid post-hoc tuning on these test sets.
