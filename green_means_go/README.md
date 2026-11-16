# Green means go — code, pre-registrations and outputs

The study ran in four pre-registered steps; every hypothesis is reported, including those that failed.

| step | pre-registration | code | results |
|---|---|---|---|
| 1 (over-call; P1) | `P1_V1_2026-10-07.md`, `preregistrations/F_STEP1_PREREG_2026-11-10.md` | `p1_overcall_job.py`, `jobs/f1_kernel.py`, `jobs/f1b_kernel.py` | `p1_v1_2026-10-07.json`, `results/f1.json`, `results/f1b.json` |
| 2 (cheap fixes) | `P1_V2_READOUT_2026-10-09.md`, `preregistrations/F_STEP2_PREREG_2026-11-10.md` | `p1_readout_job.py`, `jobs/f2_kernel.py` | `results/f2.json` |
| 3 (screening-anchored operating curves) | `preregistrations/F_STEP3_PREREG_2026-11-16.md` | `jobs/f3_job.py` | `results/f3.json`, `results/per_ecg_step3.csv` |
| 3b (confirmatory, held-out thresholds) | `preregistrations/F_STEP3B_PREREG_2026-11-16.md` | `jobs/f3_job.py` (`F3B=1`) | `results/f3b.json`, `results/per_ecg_step3b.csv` |

Figures: `figs.py` (reads the result files). Jobs ran on Kaggle (steps 1–2) and Hugging Face Jobs (steps 3–3b);
the PTB-XL and Chapman/Ningbo paths in the scripts point to our storage mount and must be changed to your copy.
