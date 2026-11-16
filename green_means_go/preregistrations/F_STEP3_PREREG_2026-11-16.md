---
status: PRE-REGISTERED 2026-11-16 (owner "go hf"), before any step-3 output exists.
platform: HF Jobs (t4-small) or Kaggle private kernel (free from Sat 2026-11-21). PTB-XL, Chapman/Ningbo, Norwegian
  athlete DB, PF12RED; ECGFounder (MIT). No athlete data are used to set any threshold, mapping or weight.
motivation: Green means go referee report R2 (Submissions/green_means_go/REVIEW_R2_2026-11-16.md), majors 1–7.
---

# F step 3: screening-anchored operating curves, stronger cheap fixes, and the Pareto frontier

## Why
Steps 1–2 showed over-call and two failed fixes at single operating points, with sensitivity measured on a broad
hospital "red" set. Clinically, the question is a trade-off: how many healthy athletes are referred, at what
sensitivity for the conditions screening exists to find. Step 3 measures that trade-off for the original model, the
two original fixes, three stronger fixes that a sports cardiologist would recognise, and the device algorithm, and
states which are on the Pareto frontier.

## Definitions (fixed now)
**Screening-relevant red set (SR-red)** — from dataset labels, the findings the International Criteria mark abnormal
and that pre-participation screening targets:
- PTB-XL SCP codes: WPW (pre-excitation); LNGQT (long QT); CLBBB (complete LBBB); IVCD; INVT and ISC_-family codes
  (T-wave inversion / ischaemic-type repolarisation: ISCAL, ISCLA, ISCIL, ISCIN, ISCAN, ISCAS, ISC_); STD_ (ST
  depression); LVH (diagnostic, i.e. not VCLVH voltage-only); and Q-wave infarction codes as a proxy for pathological
  Q waves (IMI, ILMI, IPMI, IPLMI, AMI, ASMI, ALMI, LMI, PMI).
- Chapman/Ningbo: the equivalent conditions by name (pre-excitation, QT prolongation, LBBB, IVCD, T-wave inversion /
  opposite, ST depression, LVH-with-ST-T, Q-wave infarction). The exact code list is written in the job and **printed
  before any model output is scored**; codes not present are listed as absent.
- An ECG is SR-red if it has ≥ 1 SR-red code. **Negatives** for false alarms: normal-only and green-only ECGs (as in
  steps 1–2). ECGs that are red but not SR-red are excluded from sensitivity and from false alarms (reported
  separately as "other red: flagged share").

**Athlete false-alarm (AFA)** — share flagged among Norwegian cardiologist-normal athletes (n = 26) and PF12RED
green-only recordings (156 recordings, 52 players), each reported, plus the pooled rate with a cluster bootstrap
(cluster = athlete / player, 4,000 resamples).

**Tuning data** — PTB-XL folds 1–8 only. **Test** — PTB-XL folds 9–10 and Chapman/Ningbo (sensitivity, non-athlete
false alarms), and both athlete sets (AFA). Nothing is tuned on a test set.

## Methods compared (all from the same frozen ECGFounder forward pass unless stated)
- **M0 Model output** — ECGFounder "abnormal ECG" probability (step 1–2 comparator).
- **M1 Criteria over the model's findings, screening-restricted** — step 2's rule, but the red concepts are only the
  model's concepts that correspond to SR-red (list printed before scoring), each with its own threshold = the q-th
  quantile of that concept's score on folds 1–8 normal + green-only ECGs; one shared q sets the operating point.
  Borderline pairs rule unchanged.
- **M2 Measure-then-ask hybrid** — a deterministic measurement layer applies the International Criteria to intervals
  measured with an open delineator (NeuroKit2, lead II with lead V5 fallback): heart rate < 30 bpm; PR ≥ 400 ms;
  QRS ≥ 140 ms; QTc (Bazett) ≥ 470 ms (male) / ≥ 480 ms (female; sex as recorded, male if missing) or ≤ 320 ms.
  Green measurements (bradycardia ≥ 30, PR < 400, voltage) never count. The model is asked only for morphology: the
  M1 concepts minus those replaced by measurements (long QT, LBBB/IVCD by width). ABNORMAL ⇔ any measurement rule OR
  any morphology concept over its M1 threshold (same q mechanism).
- **M3 Non-linear readout** — MLP (1 hidden layer, 256 units, dropout 0.2) on the frozen penultimate features,
  trained on folds 1–8 with criteria labels (SR-red vs normal/green; other-red excluded), early stopping on fold 8.
  **M3b** — the same head plus fine-tuning of the last backbone block (LR 1e-5, ≤ 5 epochs).
- **M4 Second model** (stretch, time-boxed to 2 h of engineering): ECG-FM open weights with a linear and an MLP head
  trained as M3. If its toolchain cannot be run in the time box, it is reported as not run.
- **M5 Device algorithm** — GE 12SL statements (Norwegian, already available) and, if present in the PF12RED
  CardioSoft XML, its interpretation statements; one fixed point (any statement outside normal/sinus/green = flag).
  Exploratory, as no sensitivity can be computed for it.
- The original Fix 1 (linear head) and Fix 2 (100-concept gate) are re-scored on the same SR-red anchor.

## Endpoints and hypotheses
For every method, operating curve = AFA (pooled and per cohort) and non-athlete green false alarms against SR-red
sensitivity on PTB-XL folds 9–10, at sensitivities 0.80 / 0.85 / 0.90 / 0.95, thresholds set on folds 1–8 to hit
each target (the realised folds 9–10 sensitivity is reported, not assumed).

- **F3-H1 (primary).** At the threshold giving 90% SR-red sensitivity on folds 1–8, M2 reduces pooled AFA by ≥ 50%
  relative to M0, with the 95% cluster-bootstrap interval of the difference excluding 0, **and** M2's realised
  folds 9–10 SR-red sensitivity is no more than 2 points below M0's.
- **F3-H2.** M2 Pareto-dominates M0: lower pooled AFA at all four sensitivity targets.
- **F3-H3.** M3 meets the F3-H1 criterion (tests whether the representation holds the information a linear head
  could not reach).
- **Transfer guard (applies to any method claimed to work).** Chapman/Ningbo SR-red sensitivity at the folds 1–8
  threshold ≥ 0.80.
- **Clinical translation.** Unnecessary referrals per 1,000 athletes screened = 1,000 × AFA at the 90% point, with
  the cluster-bootstrap interval, set beside the published false-positive rates of the International Criteria
  applied by trained readers (verified citations only).
- **Mechanism (secondary).** Logistic model of M0 flag on heart rate + green findings, within dataset, in non-athlete
  green-only and normal ECGs (PTB-XL, Chapman separately); and the heart-rate distributions behind the PTB-XL vs
  Chapman sinus-bradycardia gap.

## Disclosures fixed now
- NeuroKit2 delineation can fail on noisy ECGs; failures are counted per dataset, and a failed measurement never
  triggers a flag (it falls back to morphology only). This is reported, not tuned.
- Q-wave infarction codes in an older hospital population are a proxy for pathological Q waves in athletes.
- Brugada type 1 has no label in either non-athlete dataset and cannot be assessed.
- PF12RED marks are per player, not a structured International-Criteria reading (unchanged from steps 1–2).
- Athlete n is small (26 + 52); the pooled AFA interval will be wide. The primary hypothesis is stated on the pooled
  rate for that reason.
- Any M2 success is a property of the hybrid design with this model; it does not show ECGFounder's representation is
  athlete-aware.

## Cost
One GPU job, ~1.5–2 h on t4-small (≈ $1) or free on Kaggle after the weekly reset. M3b and M4 are the expensive parts
and are dropped first if the budget is tight.

## Log
- 2026-11-16 smoke (job 6afafb8d…): PTB-XL form statements (INVT, STD_, LNGQT) carry likelihood 0 in PTB-XL, so the steps 1–2 filter (likelihood ≥ 50) silently removed them (INVT count 0). **Implementation fix before any result was read:** these three SR-red codes are counted at any likelihood; diagnostic codes keep the ≥ 50 rule. No threshold, mapping or hypothesis changed.
- 2026-11-16 smoke 2 (job 6afb009c…, 60 per PTB-XL group, 20 per Chapman group) ran end to end (F3_DONE; numbers are plumbing only, not read as results). Chapman header scan parallelised (64 threads; 39,836 headers). Measurement success: PTB-XL 0.983, Chapman 0.925, Norwegian 1.00, PF12RED 0.994. **Full run launched (job 6afb04f35b0bff80b2ac4fa9, t4-medium).**

# RESULT (2026-11-16; HF job 6afb04f3…, 28 min; F3_DONE). Full JSON: /mnt/processed/athlete_ecg/f3/f3.json
n: tuning SR-red 2,408; PTB-XL folds 9–10 SR-red 592; Chapman SR-red 1,500; Norwegian normal 26; PF12RED 156
recordings / 52 players. Measurement success: PTB-XL 0.97, Chapman 0.94, athletes ≥ 0.99.

**Pre-registered hypotheses: all three FAIL.**
- **F3-H1 FAIL (M2 hybrid is worse).** At the 90% tuning target: AFA M2 0.582 vs M0 0.467 (difference +0.115
  [+0.016, +0.219]); sensitivity 0.892 vs 0.878. Cause, visible in the counts: the NeuroKit2 measurement layer alone
  flags 3,759 PTB-XL, 1,768 Chapman, 8/26 Norwegian and 47/156 PF12RED ECGs (median athlete QRS measured 111–112 ms;
  QRS ≥ 140 and QTc rules fire on delineation error). An open delineator is not accurate enough to carry the
  measurement half of the criteria.
- **F3-H2 FAIL** (M2 does not dominate M0).
- **F3-H3 FAIL on the sensitivity leg only.** M3 (MLP readout, criteria labels, non-athletes only) at the 90% tuning
  target: AFA **0.099** [0.039, 0.172] vs M0 0.467 (difference −0.368 [−0.458, −0.282]), but realised PTB-XL test
  sensitivity 0.840 vs 0.878 (−3.8 points > the 2-point margin). Chapman sensitivity 0.952 vs 0.953.

**Operating curves (reported as pre-registered; the matched-sensitivity reading below is post hoc).** Thresholds
set on folds 1–8 include the trained heads' own training data, so a trained head's realised test sensitivity falls
below its target (M3: 0.90 → 0.84; Fix1 linear: 0.90 → 0.81), while M0's does not. Compared at matched or higher
realised sensitivity, the trained readouts dominate the model's own output:
| method @ target | sens PTB-XL test | sens Chapman | AFA pooled [95% CI] | Norwegian | PF12RED |
|---|---|---|---|---|---|
| M0 model output @0.90 | 0.878 | 0.953 | 0.467 [0.372, 0.564] | 0.346 | 0.487 |
| M0 @0.95 | 0.914 | 0.981 | 0.604 [0.509, 0.699] | 0.577 | 0.609 |
| **M3 MLP @0.95** | **0.897** | **0.980** | **0.176 [0.102, 0.256]** | 0.038 | 0.199 |
| Fix1 linear @0.95 | 0.858 | 0.949 | 0.143 [0.085, 0.206] | 0.000 | 0.167 |
| M1 screen gate @0.90 | 0.878 | 0.975 | 0.604 | 0.692 | 0.590 |
| Fix2 100-concept gate @0.90 | 0.888 | 0.930 | 0.604 | 0.615 | 0.603 |
M3b (last-stage fine-tune) ≈ M3 (val AUROC 0.951 both). Non-athlete green false alarms fall with the trained heads
too (PTB-XL green M3 0.095–0.186 vs M0 0.355–0.479).

**Reading.** The representation does carry the green-versus-screening-red distinction: a readout trained on
non-athletes with criteria labels and screening-relevant positives, never seeing an athlete, cuts athlete false
alarms by about 3–4× at equal sensitivity. This reverses the step-1 conclusion for Fix 1, whose sensitivity was
anchored on the broad hospital red set (AF, ectopy, etc.), which forced the threshold into normal ECGs. Rules over
the model's named findings (M1, Fix2, M2) do not help: the named findings, and an open delineator, are the weak
link. Clinical units: about 470 referrals per 1,000 normal athletes for the model's own output vs about 100–180
for the criteria-trained readout at equal sensitivity (wide intervals; 26 + 52 athletes).
- **Mechanism:** flags concentrate below 50 bpm (PTB-XL 94% flagged at HR < 50 vs 14–20% at ≥ 50; Chapman 92% vs
  8–11%): a cliff at marked bradycardia, not a linear heart-rate effect. Green findings carry an odds ratio of 4.2
  (PTB-XL) and 2.2 (Chapman) beyond heart rate.
- **M5 device:** PF12RED CardioSoft XML carries only the 12SL version string ("12SL V21"), no statements, so the
  footballer device rate is unavailable; Norwegian 12SL 13/26 (0.50).
- **Next (to be pre-registered before running):** F step 3b, a confirmatory test of the trained readout with
  thresholds set on a fold the head never saw, so the sensitivity leg is tested fairly.
