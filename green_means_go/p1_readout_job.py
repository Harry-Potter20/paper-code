# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pandas", "scipy", "wfdb", "huggingface_hub", "matplotlib", "scikit-learn"]
# ///
"""P1 v2 — (a) heart-rate stratification of green over-call; (b) the READOUT FIX (Law 7: readout, not representation).

Frozen ECGFounder 12-lead backbone (MIT). Features = input to its final dense layer (forward hook). A logistic head is
retrained to predict "abnormal by athlete criteria":
  positive ("red")  = ECG with ≥ 1 finding outside the green set and outside {SR, NORM} (source-dataset labels);
  negative          = NORM/SR-only + green-only ECGs (training-related findings are NORMAL under the criteria).
Train: PTB-XL strat_fold 1–8. Test: PTB-XL fold 10 and Chapman (external).
Comparison at MATCHED SENSITIVITY on red ECGs (90%): fraction of green-only ECGs called abnormal by
  (i) ECGFounder's own ABNORMAL ECG output, (ii) a general-population head (same features, red+green = abnormal —
  the ontology public ECG datasets use), (iii) the athlete-criteria head.
Heart rate from lead II R peaks (band-passed, scipy find_peaks), for stratification (< 50, 50–59, ≥ 60 bpm).
Env: N_PER (per group per source), OUT.
Run: hf jobs uv run --flavor t4-small --timeout 3h --secrets HF_TOKEN -v hf://buckets/Chucks90/eryon-datasets:/mnt \
       --detach p1_readout_job.py"""
from __future__ import annotations

import ast
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from huggingface_hub import hf_hub_download
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, medfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = Path(os.environ.get("RAW", "/mnt/raw"))
OUT = Path(os.environ.get("OUT", "/mnt/processed/athlete_ecg/p1_v2")); OUT.mkdir(parents=True, exist_ok=True)
N_PER = int(os.environ.get("N_PER", "2500"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)
GREEN_PTB = {"SBRAD", "SARRH", "1AVB", "IRBBB", "VCLVH", "HVOLT"}
SNOMED = {"426783006": "SR", "426177001": "SB", "427393009": "SA", "270492004": "1AVB", "713426002": "IRBBB",
          "428417006": "ERV", "426995002": "JEB", "195101003": "WAVN", "164873001": "LVH"}
GREEN_CH = {"SB", "SA", "1AVB", "IRBBB", "ERV", "JEB", "WAVN"}


def log(m):
    print(f"[p1v2] {m}", flush=True)


def filter_bandpass(signal, fs):
    """Verbatim from ECGFounder util.py (MIT)."""
    b, a = iirnotch(50, 30, fs)
    f = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        f[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        f[c] = filtfilt(b, a, f[c])
    k = int(0.4 * fs) + 1; k += (k % 2 == 0)
    return f - np.stack([medfilt(f[c], kernel_size=k) for c in range(f.shape[0])])


def prep(sig):
    x = np.nan_to_num(sig.T.astype(np.float64))
    if x.shape[1] != 5000:
        x = np.stack([np.interp(np.linspace(0, x.shape[1] - 1, 5000), np.arange(x.shape[1]), c) for c in x])
    x = filter_bandpass(x, 500)
    hr = heart_rate(x[1])
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32), hr


def heart_rate(lead2, fs=500):
    s = np.abs(lead2); pk, _ = find_peaks(s, distance=int(0.3 * fs), height=np.percentile(s, 98) * 0.5)
    return float(60 * fs / np.median(np.diff(pk))) if len(pk) >= 3 else np.nan


def model():
    d = Path("/tmp/ecgf"); d.mkdir(exist_ok=True)
    for f in ("net1d.py", "tasks.txt"):
        if not (d / f).exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/{f}", d / f)
    sys.path.insert(0, str(d)); from net1d import Net1D
    labels = [l.strip() for l in open(d / "tasks.txt") if l.strip()]
    m = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
              use_bn=False, use_do=False, n_classes=len(labels))
    ck = torch.load(hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"]); m = m.to(DEV).eval()
    feats = {}
    m.dense.register_forward_hook(lambda mod, inp, out: feats.__setitem__("x", inp[0].detach()))
    return m, labels, feats


def cohort():
    items = []
    d = pd.read_csv(RAW / "ptbxl/ptbxl_database.csv")
    codes = d.scp_codes.apply(lambda s: {k for k, v in ast.literal_eval(s).items() if v >= 50 or k in GREEN_PTB | {"SR"}})
    grp = codes.apply(lambda c: "red" if c - GREEN_PTB - {"SR", "NORM"} else ("green" if c & GREEN_PTB else ("normal" if "NORM" in c else "skip")))
    for g in ("red", "green", "normal"):
        sub = d[grp == g]
        sub = sub.sample(min(len(sub), N_PER), random_state=0)
        items += [("ptbxl", g, ",".join(sorted(codes[i] & GREEN_PTB)), str(RAW / "ptbxl" / r.filename_hr), int(r.strat_fold))
                  for i, r in zip(sub.index, sub.itertuples())]
    recs = []
    for h in glob.glob(str(RAW / "ecg-arrhythmia/WFDBRecords/**/*.hea"), recursive=True):
        try:
            dx = next((l.split(":", 1)[1] for l in open(h) if l.startswith("#Dx")), "")
        except Exception:
            continue
        n = {SNOMED.get(c.strip(), "OTHER") for c in dx.split(",") if c.strip()}
        g = "red" if n - GREEN_CH - {"SR"} else ("green" if n & GREEN_CH else ("normal" if n == {"SR"} else "skip"))
        recs.append(("chapman", g, ",".join(sorted(n & GREEN_CH)), h[:-4], -1))
    for g in ("red", "green", "normal"):
        sub = [r for r in recs if r[1] == g]; RNG.shuffle(sub); items += sub[:N_PER]
    return [i for i in items if os.path.exists(i[3] + ".hea")]


@torch.no_grad()
def main():
    t0 = time.time(); m, labels, feats = model(); ia = labels.index("ABNORMAL ECG")
    items = cohort(); log(f"items {len(items)}")
    F, P, H, meta, batch, bm = [], [], [], [], [], []

    def flush():
        p = torch.sigmoid(m(torch.from_numpy(np.stack(batch)).to(DEV)))[:, ia].cpu().numpy()
        F.append(feats["x"].float().cpu().numpy()); P.extend(p.tolist()); meta.extend(bm); batch.clear(); bm.clear()
    for k, it in enumerate(items):
        try:
            sig, _ = wfdb.rdsamp(it[3])
            if sig.shape[1] != 12:
                continue
            x, hr = prep(sig); batch.append(x); bm.append(it + (hr,))
        except Exception:
            continue
        if len(batch) == 64:
            flush()
        if (k + 1) % 2000 == 0:
            log(f"{k + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    if batch:
        flush()
    X = np.concatenate(F); D = pd.DataFrame(meta, columns=["source", "group", "findings", "record", "fold", "hr"]); D["p_founder"] = P
    tr = (D.source == "ptbxl") & D.fold.between(1, 8)
    heads = {"general_population": D.group.isin(["red", "green"]).astype(int), "athlete_criteria": (D.group == "red").astype(int)}
    for name, y in heads.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)).fit(X[tr], y[tr])
        D[f"p_{name}"] = clf.predict_proba(X)[:, 1]
    D.drop(columns=["record"]).to_csv(OUT / "per_ecg.csv", index=False)
    summ = {}
    for src, test in (("ptbxl_fold10", D[(D.source == "ptbxl") & (D.fold == 10)]), ("chapman", D[D.source == "chapman"])):
        s = {"n": test.group.value_counts().to_dict()}
        for col in ("p_founder", "p_general_population", "p_athlete_criteria"):
            thr = np.quantile(test.loc[test.group == "red", col], 0.10)          # 90% sensitivity on red ECGs
            g = test[test.group == "green"]
            s[col] = {"thr_sens90": round(float(thr), 4), "green_called_abnormal": round(float((g[col] >= thr).mean()), 4),
                      "normal_called_abnormal": round(float((test.loc[test.group == "normal", col] >= thr).mean()), 4),
                      "by_hr": {b: round(float((g.loc[g.hr.between(*r), col] >= thr).mean()), 4) if g.hr.between(*r).any() else None
                                for b, r in {"<50": (0, 49.99), "50-59": (50, 59.99), ">=60": (60, 300)}.items()}}
        summ[src] = s
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("P1V2_RESULT " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
