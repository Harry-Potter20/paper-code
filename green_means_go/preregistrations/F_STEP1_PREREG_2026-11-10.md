---
status: PRE-REGISTERED 2026-11-10, before any athlete data are scored. Programme Idea F, step 1 (audit on real athletes).
  Step 2 (International-Criteria concept gate + parity constraint) gets its own prereg after the athlete label formats
  are known.
platform: Kaggle private kernel; all data public (PTB-XL PhysioNet; Norwegian Endurance Athlete ECG DB, CC BY 4.0;
  PF12RED, CC0); ECGFounder MIT, ungated.
---

# F step 1: does an ECG foundation model over-call real athletes?
P1 ("Green means go") showed ECGFounder calls *non-athletes* with athletic-type findings abnormal (PTB-XL green-only
33.7%, Chapman 17.9%, at a threshold giving 10% abnormal on normal controls). Its stated top limitation: "these are
non-athletes, not athletes". This step closes that gap.

**Pipeline:** identical to `CV_Research/athlete_ecg/p1_overcall_job.py`:
- ECGFounder 12-lead, its own ABNORMAL ECG output;
- ECGFounder bandpass + global z-score;
- 5,000 samples at 500 Hz.

**Threshold:** recomputed on PTB-XL NORM + SR controls (P1's definition; up to 2,000) as the 90th percentile of
P(ABNORMAL), so 10% of normal controls are called abnormal.

**Sanity replication:** PTB-XL green-only over-call must land within ±5 points of P1's 33.7%. If it does not, the run
is a pipeline error and is not interpreted.

**Athlete cohorts:**
- **Norwegian Endurance Athlete ECG DB:** 28 elite athletes, cardiologist-read per the International Criteria; labels
  parsed from the WFDB header comments.
- **PF12RED:** 54 La Liga players, 163 recordings; labels from LabelData.xlsx; MUSE-style XML waveforms.

The parsing of both label formats is reported verbatim before any rate is computed.

**Primary analysis set:** athletes whose cardiologist read is normal, or training-related findings only ("green"),
per the dataset labels. If a cohort's labels cannot be mapped to normal / green / abnormal, report the cohort-wide
rate only and say so.

**Hypothesis H-F0:** in the primary set, ECGFounder flags ≥ 20% of athletes as abnormal at the control-calibrated
threshold (P1's non-athlete range was 18–34%). Wilson CI per cohort. Reported whichever way it falls.

**Secondary (descriptive):**
- the top non-normal ECGFounder labels fired on flagged athletes, i.e. what the model thinks they have;
- over-call vs heart rate (< 50 bpm vs ≥ 50 bpm).

**Small-n honesty:** 28 + ~54 subjects. CIs will be wide; per-recording rates in PF12RED are clustered by player, so we
use cluster-robust (player-level bootstrap) CIs.

## RESULT (Kaggle `f1-athlete-ecg-audit` v2, 2026-11-10; kernel deleted, outputs kept locally)
- **Sanity:** PTB-XL green-only over-call 33.6% (P1 33.7%), control-calibrated threshold 0.9845. **Replicates.**
- **Norwegian elite athletes (primary set = cardiologist "Normal ECG"; mapping from the C: label only, blind to model
  output):** 26 normal, 2 borderline.
  - **ECGFounder flags 9 / 26 normal athletes abnormal = 34.6% [Wilson 19.4, 53.8].**
  - **H-F0 (≥ 20%) is met on the point estimate**; the CI lower bound (19.4%) touches the bar. Both borderline
    athletes are flagged.
  - What it thinks they have: sinus bradycardia ×7, marked sinus bradycardia ×6, right axis deviation ×2, left atrial
    enlargement ×2, **anterior infarct ×1**.
  - **Conventional comparator:** the device's own GE 12SL algorithm labels 13 / 26 cardiologist-normal athletes
    borderline (7) or abnormal (6), i.e. 50%. The FM is not uniquely bad; automated interpretation over-calls athletes
    generally.
- **PF12RED:** 0 / 163 parsed. The files are GE CardioSoft XML, not MUSE; the parser is to be fixed (step 1b).
  - LabelData.xlsx gives per-player findings marked "X": SR, SB, iRBBB, negative T-wave per lead; plus race
    (Caucasian 40, Latin 7, African 6, Asian 1).
  - That enables a small ancestry check later.
- **Step 1b mapping rule (fixed now, before parsing).** "Green-only" player = no negative T-wave marked in I, II, aVL,
  aVF or V3–V6. Negative T in III, aVR and V1 (and V2) is normal under the International Criteria; V2–V4 TWI preceded
  by J-elevation in Black athletes cannot be checked from the labels, so those cases are excluded from the primary set.

## STEP 1b RESULT: PF12RED (Kaggle `f1b-pf12red`; GE CardioSoft parser; 162 / 163 parsed; kernel deleted)
- Threshold fixed from step 1 (0.9845); player-level bootstrap.
- **Primary set:** green-only players per the pre-specified label rule (52 / 54).
  - **ECGFounder flags 47.4% of green-only recordings abnormal [36.4, 58.5].**
  - **H-F0 met** (CI entirely above 20%).
- **All players:** 48.8% [37.7, 59.0].
- **By ancestry, green-only (tiny n, descriptive only):** African 60% (5 players), Caucasian 48.8% (40), Latin 23.5%
  (6), Asian 1 player.
- **What it calls them:** sinus bradycardia (61), marked sinus bradycardia (50), LVH (14), **lateral infarct (11)**,
  RBBB (10), incomplete RBBB (8).
- **Across both real-athlete cohorts:** Norwegian elite 34.6%, La Liga 47.4%, vs 10% on general-population normal
  controls by construction. ECG foundation models over-call real athletes at 3.5–4.7× the control rate. Infarct labels
  appear in both cohorts.
- **Next: step 2 prereg.** The International-Criteria concept gate, with an ancestry false-positive parity
  constraint. PF12RED's per-lead labels (SB, iRBBB, NTW by lead) make it the gate's natural external test.
