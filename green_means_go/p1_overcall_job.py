# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pandas", "scipy", "wfdb", "huggingface_hub", "matplotlib"]
# ///
"""P1 (RESEARCH_PROGRAM_2026Q4.md) — "Green means go": does ECG AI call training-related ("green") findings abnormal?

Model: ECGFounder 12-lead (PKUDigitalHealth, MIT; HF weights + net1d.py/util.py from its GitHub at runtime), its own
150-label sigmoid head incl. NORMAL ECG / BORDERLINE ECG / ABNORMAL ECG. Preprocessing as its dataset.py:
filter_bandpass(500 Hz) then global z-score; 10 s at 500 Hz = 5000 samples.
Cohorts (labels from the source datasets, never from the model):
  PTB-XL   green-only: scp codes ⊆ {SR, NORM, SBRAD, SARRH, 1AVB, IRBBB, VCLVH, HVOLT} with ≥ 1 green finding;
           controls: NORM + SR only (random, same n).
  Chapman  (raw/ecg-arrhythmia, 45k, SNOMED): green-only Dx ⊆ {SR, SB, SA, 1AVB, IRBBB, ERV, JEB, WAVN} with ≥ 1 green;
           LVH (164873001) analysed separately (voltage-only vs not unknown); controls: SR only.
Endpoints: P(ABNORMAL ECG); rate above 0.5 and above the threshold giving 10% "abnormal" on matched controls;
per green finding; top non-green labels fired on green-only ECGs (what the model thinks they are).
Writes per-ECG CSV + summary JSON. Env: N_MAX (per cohort), OUT.
Run: hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN -v hf://buckets/Chucks90/eryon-datasets:/mnt \
       --detach p1_overcall_job.py"""
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

RAW = Path(os.environ.get("RAW", "/mnt/raw"))
OUT = Path(os.environ.get("OUT", "/mnt/processed/athlete_ecg/p1_v1")); OUT.mkdir(parents=True, exist_ok=True)
N_MAX = int(os.environ.get("N_MAX", "2000"))
GH = "https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/{}"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)
GREEN_PTB = {"SBRAD", "SARRH", "1AVB", "IRBBB", "VCLVH", "HVOLT"}
SNOMED = {"426783006": "SR", "426177001": "SB", "427393009": "SA", "270492004": "1AVB", "713426002": "IRBBB",
          "428417006": "ERV", "426995002": "JEB", "195101003": "WAVN", "164873001": "LVH"}
GREEN_CH = {"SB", "SA", "1AVB", "IRBBB", "ERV", "JEB", "WAVN"}


def log(m):
    print(f"[p1] {m}", flush=True)


def fetch_code():
    d = Path("/tmp/ecgf"); d.mkdir(exist_ok=True)
    for f in ("net1d.py", "tasks.txt"):
        if not (d / f).exists():
            urllib.request.urlretrieve(GH.format(f), d / f)
    sys.path.insert(0, str(d))
    return [l.strip() for l in open(d / "tasks.txt") if l.strip()]


def model(n):
    from net1d import Net1D
    m = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
              use_bn=False, use_do=False, n_classes=n)
    ck = torch.load(hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth"), map_location="cpu", weights_only=False)
    miss = m.load_state_dict(ck["state_dict"], strict=False)
    log(f"weights: missing {miss.missing_keys[:4]} unexpected {miss.unexpected_keys[:4]}")
    return m.to(DEV).eval()


def filter_bandpass(signal, fs):
    """Verbatim from ECGFounder util.py (MIT): 50 Hz notch, 0.67–40 Hz Butterworth, median baseline removal."""
    from scipy.signal import butter, filtfilt, iirnotch, medfilt
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    return filtered_signal - baseline


def prep(sig):
    x = np.nan_to_num(sig.T.astype(np.float64))                      # (12, T)
    if x.shape[1] != 5000:
        x = np.stack([np.interp(np.linspace(0, x.shape[1] - 1, 5000), np.arange(x.shape[1]), c) for c in x])
    x = filter_bandpass(x, 500)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def ptbxl():
    d = pd.read_csv(RAW / "ptbxl/ptbxl_database.csv")
    codes = d.scp_codes.apply(lambda s: {k for k, v in ast.literal_eval(s).items() if v >= 50 or k in GREEN_PTB | {"SR"}})
    green = codes.apply(lambda c: bool(c) and c <= GREEN_PTB | {"SR", "NORM"} and bool(c & GREEN_PTB))
    norm = codes.apply(lambda c: c <= {"SR", "NORM"} and "NORM" in c)
    g = d[green].assign(findings=codes[green].apply(lambda c: ",".join(sorted(c & GREEN_PTB))), group="green")
    n = d[norm].sample(min(len(g), norm.sum()), random_state=0).assign(findings="", group="control")
    out = pd.concat([g, n]).head(2 * N_MAX)
    return [("ptbxl", r.group, r.findings, str(RAW / "ptbxl" / r.filename_hr), int(r.strat_fold), float(r.age)) for r in out.itertuples()]


def chapman():
    root = RAW / "ecg-arrhythmia/WFDBRecords"
    recs = []
    for h in glob.glob(str(root / "**" / "*.hea"), recursive=True):
        try:
            dx = next((l.split(":", 1)[1] for l in open(h) if l.startswith("#Dx")), "")
            age = next((l.split(":", 1)[1].strip() for l in open(h) if l.startswith("#Age")), "nan")
        except Exception:
            continue
        codes = {c.strip() for c in dx.split(",") if c.strip()}
        names = {SNOMED.get(c, "OTHER:" + c) for c in codes}
        recs.append((h[:-4], names, age))
    log(f"chapman headers: {len(recs)}")
    green = [(p, n, a) for p, n, a in recs if n and n <= GREEN_CH | {"SR"} and n & GREEN_CH]
    lvh = [(p, n, a) for p, n, a in recs if n <= {"SR", "LVH"} and "LVH" in n]
    ctrl = [(p, n, a) for p, n, a in recs if n == {"SR"}]
    RNG.shuffle(green); RNG.shuffle(ctrl)
    out = [("chapman", "green", ",".join(sorted(n & GREEN_CH)), p, -1, a) for p, n, a in green[:N_MAX]]
    out += [("chapman", "lvh_only", "LVH", p, -1, a) for p, n, a in lvh[:N_MAX]]
    out += [("chapman", "control", "", p, -1, a) for p, n, a in ctrl[:len(out)]]
    return out


@torch.no_grad()
def main():
    t0 = time.time(); labels = fetch_code(); m = model(len(labels))
    ia, ib, inn = labels.index("ABNORMAL ECG"), labels.index("BORDERLINE ECG"), labels.index("NORMAL ECG")
    items = [i for i in ptbxl() + chapman() if os.path.exists(i[3] + ".hea")]
    log(f"items {len(items)}")
    recs, batch, meta = [], [], []
    nerr = [0]

    def flush():
        p = torch.sigmoid(m(torch.from_numpy(np.stack(batch)).to(DEV))).cpu().numpy()
        for (src, grp, fnd, path, fold, age), pr in zip(meta, p):
            top = [labels[i] for i in np.argsort(-pr)[:5]]
            recs.append({"source": src, "group": grp, "findings": fnd, "record": path, "fold": fold, "age": age,
                         "p_abnormal": float(pr[ia]), "p_borderline": float(pr[ib]), "p_normal": float(pr[inn]),
                         "top5": "|".join(top), "probs": json.dumps({labels[i]: round(float(pr[i]), 4) for i in np.where(pr > 0.3)[0]})})
        batch.clear(); meta.clear()
    for k, it in enumerate(items):
        try:
            sig, _ = wfdb.rdsamp(it[3])
            if sig.shape[1] != 12:
                continue
            batch.append(prep(sig)); meta.append(it)
        except Exception as e:
            nerr[0] += 1
            if nerr[0] <= 3:
                log(f"read/prep fail {it[3]}: {type(e).__name__}: {str(e)[:150]}")
            continue
        if len(batch) == 64:
            flush()
        if (k + 1) % 1000 == 0:
            log(f"{k + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    if batch:
        flush()
    D = pd.DataFrame(recs); D.to_csv(OUT / "per_ecg.csv", index=False)
    summ = {}
    for src, g in D.groupby("source"):
        c = g[g.group == "control"]
        thr = float(np.quantile(c.p_abnormal, 0.90)) if len(c) else 0.5
        s = {"thr_at_10pct_control": round(thr, 4), "groups": {}}
        for grp, h in g.groupby("group"):
            s["groups"][grp] = {"n": len(h), "mean_p_abnormal": round(h.p_abnormal.mean(), 4),
                                "rate_p_abn_gt_0.5": round((h.p_abnormal > 0.5).mean(), 4), "rate_above_ctrl_thr": round((h.p_abnormal > thr).mean(), 4)}
        gr = g[g.group == "green"]
        for f in sorted({x for fs in gr.findings for x in fs.split(",") if x}):
            h = gr[gr.findings.str.split(",").apply(lambda l: f in l)]
            s["groups"][f"green:{f}"] = {"n": len(h), "rate_p_abn_gt_0.5": round((h.p_abnormal > 0.5).mean(), 4),
                                         "rate_above_ctrl_thr": round((h.p_abnormal > thr).mean(), 4)}
        summ[src] = s
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("P1_RESULT " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
