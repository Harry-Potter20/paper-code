---
status: PRE-REGISTERED 2026-11-16 (owner instruction), before any step-3b output exists.
platform: HF Jobs (t4-medium). Same data, preprocessing, SR-red definition and athlete sets as step 3.
motivation: step 3 (F_STEP3_PREREG_2026-11-16.md) found, post hoc, that a criteria-trained readout cuts athlete false
  alarms about 3–4× at matched sensitivity, but its pre-registered test (F3-H3) failed on sensitivity because its
  thresholds were set on folds that included its own training data. Step 3b is the fair, confirmatory test.
---

# F step 3b: confirmatory test of the criteria-trained readout with held-out thresholds

## Design (fixed now)
- **Readout (M3, unchanged architecture):** MLP on frozen ECGFounder penultimate features (1 hidden layer, 256 units,
  dropout 0.2), labels SR-red vs normal/green-only (other-red excluded), as in step 3.
- **Splits (PTB-XL strat_fold):** train folds 1–6; early stopping on fold 7 (AUROC); **thresholds on fold 8 only**
  (never used for training or early stopping); test folds 9–10, Chapman/Ningbo, Norwegian normal athletes (n = 26),
  PF12RED green-only (156 recordings, 52 players).
- **Comparator (M0):** ECGFounder's own "abnormal ECG" output, threshold also set on fold 8.
- **Secondary:** the linear criteria head (logistic, same labels and splits as M3) and M3 retrained with seeds 1 and
  2 (robustness).
- **Operating point:** the threshold giving 90% SR-red sensitivity on fold 8 (primary); 95% reported.

## Hypothesis (primary, seed 0)
**F3b-H1.** At the fold-8 90% threshold: pooled athlete false-alarm rate M3 ≤ 0.5 × M0, with the 95% cluster-
bootstrap interval of the difference (M3 − M0) below 0, **and** M3's realised PTB-XL folds 9–10 SR-red sensitivity ≥
M0's − 0.02.
- **Transfer guard:** M3's Chapman/Ningbo SR-red sensitivity ≥ 0.80.
- **Robustness (reported, not part of the pass rule):** F3b-H1 evaluated for seeds 1 and 2; "robust" if all three
  seeds pass.

## Consequence (fixed now)
- PASS → the paper is rewritten around "the over-call problem, and a criteria-labelled readout that fixes it", with
  step 3's post hoc table as supporting evidence and step 3b as the confirmatory result.
- FAIL → the paper keeps the over-call finding as its headline and reports the readout as promising but unconfirmed.

## Disclosures
- Fold 8 contains about 300 SR-red ECGs, so the 90% threshold carries sampling noise; the realised test sensitivity
  is reported, not assumed.
- Same data as step 3; this is a re-split, not new data. The athlete sets are small (26 + 52).

## Log
- 2026-11-16: implemented as F3B=1 mode of f3_job.py (step3b()); launched HF job 6afb169ff7633f04f3557593 (t4-medium).

# RESULT (2026-11-16; HF job 6afb169f…, 28 min; F3B_DONE) — **F3b-H1 PASS, robust across 3 seeds**
Thresholds from fold 8 (299 SR-red ECGs); test PTB-XL folds 9–10 (592 SR-red), Chapman (1,500 SR-red), athletes 26 +
52 players (182 ECGs).
| method @ fold-8 90% | sens PTB-XL test | sens Chapman | athlete false alarms, pooled [95% CI] | PTB-XL green | PTB-XL normal |
|---|---|---|---|---|---|
| M0 model output | 0.894 | 0.971 | 0.528 [0.434, 0.624] | 0.414 | 0.128 |
| **M3 readout, seed 0 (primary)** | **0.875** | **0.981** | **0.176 [0.103, 0.254]** | 0.157 | 0.066 |
| M3 seed 1 | 0.878 | 0.975 | 0.159 [0.090, 0.235] | 0.145 | 0.066 |
| M3 seed 2 | 0.880 | 0.986 | 0.220 [0.137, 0.305] | 0.166 | 0.076 |
| linear criteria head | 0.900 | 0.951 | 0.264 [0.184, 0.351] | 0.234 | 0.184 |
- Primary: AFA 0.176 vs 0.528 (ratio 0.33 ≤ 0.5), difference −0.352 [−0.440, −0.268]; sensitivity 0.875 vs 0.894
  (−1.9 points, within the 2-point margin); Chapman 0.981 ≥ 0.80. **PASS.** Seeds 1 and 2 also pass (**robust**).
- At the 95% point: M3 sensitivity 0.949 vs M0 0.922, athlete false alarms 0.357 vs 0.626 (still lower at higher
  sensitivity).
- Consequence (pre-registered): the paper is rewritten around the over-call problem and the criteria-labelled
  readout that fixes it.
