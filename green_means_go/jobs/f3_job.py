# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy<2.3", "pandas", "scipy", "wfdb", "huggingface_hub", "scikit-learn", "openpyxl", "neurokit2", "PyWavelets"]
# ///
"""Programme Idea F step 3 (F_STEP3_PREREG_2026-11-16.md): screening-anchored operating curves, stronger cheap fixes,
Pareto frontier. Preprocessing, cohorts, PF12RED parser and step-2 concept lists are copied verbatim from
kaggle_f2/f2_kernel.py and CV_Research/athlete_ecg/p1_readout_job.py.
Tuning: PTB-XL folds 1-8 only. Test: PTB-XL folds 9-10 + Chapman/Ningbo (sensitivity, non-athlete false alarms) and
the Norwegian cardiologist-normal athletes + PF12RED green-only recordings (athlete false alarms, AFA).
Methods: M0 model abnormal output; Fix1 original linear head (broad red); Fix2 original 100-concept gate;
M1 screening-restricted criteria gate with per-concept quantile thresholds; M2 measure-then-ask hybrid (NeuroKit2
intervals + M1 morphology); M3 MLP readout; M3b MLP + last-stage fine-tune; M5 GE device statement (exploratory).
M4 (ECG-FM) is not run in this job (reported as not run).
Env: SMOKE=1 (small samples), OUT (/mnt/processed/athlete_ecg/f3), RAW (/mnt/raw), NPROC (cpu count).
Run: hf jobs uv run --flavor t4-medium --timeout 4h --secrets-file <f> -v hf://buckets/Chucks90/eryon-datasets:/mnt f3_job.py"""
import ast, base64, glob, io, json, os, re, sys, time, urllib.request, zipfile, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd, torch, wfdb
from huggingface_hub import hf_hub_download
from scipy.signal import butter, filtfilt, iirnotch, medfilt

warnings.filterwarnings("ignore")
SMOKE = os.environ.get("SMOKE") == "1"
RAW = Path(os.environ.get("RAW", "/mnt/raw")); OUT = Path(os.environ.get("OUT", "/mnt/processed/athlete_ecg/f3" + ("_smoke" if SMOKE else "")))
OUT.mkdir(parents=True, exist_ok=True); TMP = Path("/tmp/f3"); TMP.mkdir(exist_ok=True)
NPROC = int(os.environ.get("NPROC", str(os.cpu_count() or 4)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; T0 = time.time()
def log(m): print(f"[f3 {time.time() - T0:6.0f}s] {m}", flush=True)
TARGETS = [0.80, 0.85, 0.90, 0.95]
N_PTB = {"sr_red": 3000, "green": 2500, "normal": 3000, "other_red": 1500}; N_CH = {"sr_red": 1500, "green": 1500, "normal": 1500, "other_red": 500}
if SMOKE:
    N_PTB = {k: 60 for k in N_PTB}; N_CH = {k: 20 for k in N_CH}

# ---------------------------------------------------------------- label definitions (fixed in the prereg)
GREEN_PTB = {"SBRAD", "SARRH", "1AVB", "IRBBB", "VCLVH", "HVOLT"}
SR_PTB = {"WPW", "LNGQT", "CLBBB", "IVCD", "INVT", "ISCAL", "ISCLA", "ISCIL", "ISCIN", "ISCAN", "ISCAS", "ISC_", "STD_", "LVH",
          "IMI", "ILMI", "IPMI", "IPLMI", "AMI", "ASMI", "ALMI", "LMI", "PMI"}
SNOMED_GREEN = {"426177001": "SB", "427393009": "SA", "270492004": "1AVB", "713426002": "IRBBB", "428417006": "ERV", "426995002": "JEB", "195101003": "WAVN"}
SNOMED_SR = {"74390002": "WPW", "111975006": "LNGQT", "164909002": "LBBB", "733534002": "CLBBB", "698252002": "IVCD", "59931005": "TINV",
             "429622005": "STD", "164917005": "QAB", "164865005": "MI", "164867002": "OLDMI", "54329005": "AMI"}
SR_CODE = "426783006"
# ECGFounder concepts (names verbatim from tasks.txt)
SR_MORPH = ["T WAVE INVERSION NOW EVIDENT IN", "T WAVE INVERSION MORE EVIDENT IN", "INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN",
            "WITH REPOLARIZATION ABNORMALITY", "ST NOW DEPRESSED IN", "ST MORE DEPRESSED IN",
            "SEPTAL INFARCT", "ANTERIOR INFARCT", "LATERAL INFARCT", "ANTEROSEPTAL INFARCT", "ANTEROLATERAL INFARCT", "INFERIOR INFARCT",
            "INFERIOR-POSTERIOR INFARCT", "POSTERIOR INFARCT", "WOLFF-PARKINSON-WHITE", "WITH SHORT PR", "LEFT VENTRICULAR HYPERTROPHY"]
SR_WIDTH_QT = ["LEFT BUNDLE BRANCH BLOCK", "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY", "NONSPECIFIC INTRAVENTRICULAR BLOCK", "WITH QRS WIDENING",
               "WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY", "BIFASCICULAR BLOCK", "PROLONGED QT", "QT HAS LENGTHENED"]   # M1 only; M2 measures these
# step-2 sets (verbatim)
GREEN = {"SINUS BRADYCARDIA", "MARKED SINUS BRADYCARDIA", "WITH SINUS ARRHYTHMIA", "WITH MARKED SINUS ARRHYTHMIA", "WITH 1ST DEGREE AV BLOCK",
         "WITH PROLONGED AV CONDUCTION", "WITH 2ND DEGREE AV BLOCK MOBITZ I", "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK", "RSR' PATTERN IN V1",
         "RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY", "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY",
         "EARLY REPOLARIZATION", "ECTOPIC ATRIAL RHYTHM", "JUNCTIONAL RHYTHM", "JUNCTIONAL BRADYCARDIA", "WITH JUNCTIONAL ESCAPE COMPLEXES"}
BORDER = {"LEFT AXIS DEVIATION", "ABNORMAL LEFT AXIS DEVIATION", "LEFTWARD AXIS", "RIGHT AXIS DEVIATION", "ABNORMAL RIGHT AXIS DEVIATION",
          "RIGHTWARD AXIS", "RIGHT SUPERIOR AXIS DEVIATION", "LEFT ANTERIOR FASCICULAR BLOCK", "LEFT ATRIAL ENLARGEMENT", "RIGHT ATRIAL ENLARGEMENT",
          "BIATRIAL ENLARGEMENT", "RIGHT BUNDLE BRANCH BLOCK"}
IGNORE = {"ABNORMAL ECG", "NORMAL ECG", "BORDERLINE ECG", "otherwise normal ecg", "NORMAL SINUS RHYTHM", "SINUS RHYTHM", "SINUS TACHYCARDIA",
          "PREMATURE ATRIAL COMPLEXES", "PREMATURE SUPRAVENTRICULAR COMPLEXES", "SUPRAVENTRICULAR COMPLEXES", "LOW VOLTAGE QRS", "PEDIATRIC ECG ANALYSIS",
          "QT HAS SHORTENED", "T WAVE AMPLITUDE HAS DECREASED IN", "T WAVE AMPLITUDE HAS INCREASED IN", "ST NO LONGER DEPRESSED IN", "ST NO LONGER ELEVATED IN",
          "ST LESS DEPRESSED IN", "ST LESS ELEVATED IN", "NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN", "T WAVE INVERSION NO LONGER EVIDENT IN",
          "T WAVE INVERSION LESS EVIDENT IN", "ANTEROLATERAL LEADS", "R IN AVL", "ACUTE", "BLOCKED", "MASKED BY FASCICULAR BLOCK", "SINUS/ATRIAL CAPTURE",
          "WITH RETROGRADE CONDUCTION", "OR DIGITALIS EFFECT", "FUSION COMPLEXES", "PULMONARY DISEASE PATTERN"}
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ---------------------------------------------------------------- preprocessing (verbatim) + measurements
def filter_bandpass(signal, fs):
    b, a = iirnotch(50, 30, fs); f = np.zeros_like(signal)
    for c in range(signal.shape[0]): f[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]): f[c] = filtfilt(b, a, f[c])
    k = int(0.4 * fs) + 1; k += (k % 2 == 0)
    return f - np.stack([medfilt(f[c], kernel_size=k) for c in range(f.shape[0])])


def reorder(sig, names):
    names = [n.upper().replace("-", "") for n in names]; idx = {n: i for i, n in enumerate(names)}
    if all(l in idx for l in LEADS): return sig[:, [idx[l] for l in LEADS]]
    if all(l in idx for l in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")):
        I, II = sig[:, idx["I"]], sig[:, idx["II"]]; III = II - I
        return np.stack([I, II, III, -(I + II) / 2, I - II / 2, II - I / 2] + [sig[:, idx[f"V{k}"]] for k in range(1, 7)], 1)
    raise ValueError(f"leads {names}")


def measure(lead, fs=500):
    """International-Criteria intervals from one lead with NeuroKit2 (DWT delineation). NaN where implausible / failed."""
    import neurokit2 as nk
    out = {"hr": np.nan, "pr": np.nan, "qrs": np.nan, "qt": np.nan}
    try:
        _, info = nk.ecg_peaks(nk.ecg_clean(lead, sampling_rate=fs), sampling_rate=fs); r = np.asarray(info["ECG_R_Peaks"])
        if len(r) < 4: return out
        rr = np.median(np.diff(r)) / fs; out["hr"] = 60 / rr if 0.3 < rr < 3 else np.nan
        _, w = nk.ecg_delineate(lead, r, sampling_rate=fs, method="dwt")
        g = lambda k: np.asarray(w.get(k, []), float)
        po, ro, rf, tf = g("ECG_P_Onsets"), g("ECG_R_Onsets"), g("ECG_R_Offsets"), g("ECG_T_Offsets")
        n = min(len(po), len(ro), len(rf), len(tf))
        med = lambda a: float(np.nanmedian(a)) * 1000 / fs if np.isfinite(a).any() else np.nan
        pr, qrs, qt = med(ro[:n] - po[:n]), med(rf[:n] - ro[:n]), med(tf[:n] - ro[:n])
        out["pr"] = pr if 60 <= pr <= 600 else np.nan; out["qrs"] = qrs if 40 <= qrs <= 250 else np.nan; out["qt"] = qt if 200 <= qt <= 700 else np.nan
    except Exception:
        pass
    return out


def prep_and_measure(sig, fs):
    """sig (T, 12) standard order -> (model input 12x5000 float32, measurements)."""
    x = np.nan_to_num(sig.T.astype(np.float64)); n10 = int(10 * fs)
    x = x[:, :n10] if x.shape[1] >= n10 else np.pad(x, ((0, 0), (0, n10 - x.shape[1])))
    if x.shape[1] != 5000:
        x = np.stack([np.interp(np.linspace(0, x.shape[1] - 1, 5000), np.arange(x.shape[1]), c) for c in x])
    x = filter_bandpass(x, 500)
    m = measure(x[1])
    if not np.isfinite([m["pr"], m["qrs"], m["qt"]]).all():                 # lead V5 fallback
        m5 = measure(x[10]); m = {k: (m[k] if np.isfinite(m[k]) else m5[k]) for k in m}
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32), m


def load_item(it):
    src, grp, path, fold, sex, cluster, cohort = it
    try:
        if src in ("ptbxl", "chapman", "norwegian"):
            s_, fd = wfdb.rdsamp(path); s_ = reorder(s_, fd["sig_name"]) if src == "norwegian" else s_
            if s_.shape[1] != 12: return None
            x, m = prep_and_measure(s_, fd["fs"])
        else:
            return None
        return (x, m, it)
    except Exception:
        return None


# ---------------------------------------------------------------- cohorts
def ptbxl_items():
    d = pd.read_csv(RAW / "ptbxl/ptbxl_database.csv")
    FORM_SR = {"INVT", "STD_", "LNGQT"}                         # PTB-XL form statements carry likelihood 0: count them at any likelihood
    codes = d.scp_codes.apply(lambda s: {k for k, v in ast.literal_eval(s).items() if v >= 50 or k in GREEN_PTB | {"SR"} | FORM_SR})
    grp = codes.apply(lambda c: "sr_red" if c & SR_PTB else ("other_red" if c - GREEN_PTB - {"SR", "NORM"} else
                                                            ("green" if c & GREEN_PTB else ("normal" if "NORM" in c else "skip"))))
    d = d.assign(group=grp); out = []
    log("PTB-XL groups (all): " + json.dumps(d.group.value_counts().to_dict()))
    log("PTB-XL SR-red code counts: " + json.dumps({k: int(codes.apply(lambda c: k in c).sum()) for k in sorted(SR_PTB)}))
    for g, n in N_PTB.items():
        sub = d[d.group == g]; sub = sub.sample(min(len(sub), n), random_state=0)
        out += [("ptbxl", g, str(RAW / "ptbxl" / r.filename_hr), int(r.strat_fold), "F" if r.sex == 1 else "M", f"ptb{r.ecg_id}", "ptbxl") for r in sub.itertuples()]
    return out


def _read_head(h):
    try:
        return h, open(h).read().splitlines()
    except Exception:
        return h, None


def chapman_items():
    recs = []; rng = np.random.default_rng(0)
    from concurrent.futures import ThreadPoolExecutor              # header reads over the bucket mount are latency-bound
    hs = sorted(glob.glob(str(RAW / "ecg-arrhythmia/WFDBRecords/*/*/*.hea"))); log(f"Chapman headers {len(hs)}")
    with ThreadPoolExecutor(64) as ex:
        heads = list(ex.map(_read_head, hs))
    for h, lines in heads:
        if lines is None:
            continue
        dx = next((l.split(":", 1)[1] for l in lines if l.startswith("#Dx")), ""); sx = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("#Sex")), "")
        c = {x.strip() for x in dx.split(",") if x.strip()}
        g = ("sr_red" if c & set(SNOMED_SR) else ("other_red" if c - set(SNOMED_GREEN) - {SR_CODE} else
             ("green" if c & set(SNOMED_GREEN) else ("normal" if c == {SR_CODE} else "skip"))))
        recs.append(("chapman", g, h[:-4], -1, "F" if sx.lower().startswith("f") else "M", "ch" + Path(h).stem, "chapman", tuple(sorted(c & set(SNOMED_SR)))))
    log("Chapman groups (all): " + json.dumps(pd.Series([r[1] for r in recs]).value_counts().to_dict()))
    cnt = {}
    for r in recs:
        for k in r[7]: cnt[SNOMED_SR[k]] = cnt.get(SNOMED_SR[k], 0) + 1
    log("Chapman SR-red code counts (absent codes not listed): " + json.dumps(cnt))
    out = []
    for g, n in N_CH.items():
        sub = [r[:7] for r in recs if r[1] == g]; rng.shuffle(sub); out += sub[:n]
    return out


def norwegian_items():
    base = "https://physionet.org/files/norwegian-athlete-ecg/1.0.0/"
    recs = urllib.request.urlopen(base + "RECORDS", timeout=60).read().decode().split(); out, sl12 = [], {}
    for r in recs:
        for ext in (".hea", ".dat"):
            p = TMP / f"{r}{ext}"
            if not p.exists(): urllib.request.urlretrieve(base + r + ext, p)
        com = wfdb.rdheader(str(TMP / r)).comments
        card = " ".join(c for c in com if c.strip().startswith("C:")); dev = " ".join(c for c in com if c.strip().startswith("SL12:"))
        grp = "normal" if "Normal ECG" in card else "other"; sl12[f"nor_{r}"] = ("Borderline ECG" in dev or "Abnormal ECG" in dev)
        out.append(("norwegian", grp, str(TMP / r), -1, "M", f"nor_{r}", "norwegian"))
    return out, sl12


def parse_cardiosoft(xml_bytes):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes); strip = root.find(".//StripData")
    if strip is None: raise ValueError("no StripData")
    fs = float((strip.findtext("SampleRate") or "500").strip()); res = float((strip.findtext("Resolution") or "1").strip()); leads = {}
    for wd in strip.iter("WaveformData"):
        lid = wd.get("lead"); txt = (wd.text or "").strip()
        if lid and txt: leads[lid] = np.array([float(v) for v in re.split(r"[,\s]+", txt) if v], np.float64) * res / 1000.0
    if len(leads) < 8: raise ValueError(f"leads {list(leads)}")
    L = min(len(v) for v in leads.values()); names = list(leads)
    stm = [re.sub(r"\s+", " ", (e.text or "")).strip() for e in root.iter() if re.search(r"statement|diagnosis|interpretation", e.tag, re.I) and (e.text or "").strip()]
    return np.stack([leads[n][:L] for n in names], 1), names, int(fs), stm


def pf12red():
    z = TMP / "pf.zip"
    if not z.exists(): urllib.request.urlretrieve("https://codeload.github.com/dradolfomunoz/PF12RED/zip/refs/heads/main", z)
    zf = zipfile.ZipFile(z); names = zf.namelist()
    lab = pd.read_excel(io.BytesIO(zf.read(next(n for n in names if n.endswith("LabelData.xlsx")))))
    hdr = lab.iloc[0].tolist(); cols = [str(h) if str(h) != "nan" else str(c) for h, c in zip(hdr, lab.columns)]
    L = lab.iloc[1:].copy(); L.columns = cols; L = L[pd.to_numeric(L[cols[0]], errors="coerce").notna()]
    TW_BAD = ["I", "II", "aVL", "aVF", "V3", "V4", "V5", "V6"]
    green = {str(int(float(r[cols[0]]))) for _, r in L.iterrows() if not any(str(r.get(ld, "")).strip().upper() == "X" for ld in TW_BAD if ld in L.columns)}
    log(f"PF12RED label columns: {cols}")
    X, M, meta, stmts = [], [], [], {}
    for n in sorted(n for n in names if n.lower().endswith(".xml")):
        pid = n.split("/")[-1].split("_")[0]
        if pid not in green: continue
        try:
            s_, ln, fs, stm = parse_cardiosoft(zf.read(n)); x, m = prep_and_measure(reorder(s_, ln), fs)
        except Exception:
            continue
        X.append(x); M.append(m); meta.append(("pf12red", "green", n, -1, "M", f"pf_{pid}", "pf12red")); stmts[n] = stm
    return X, M, meta, stmts


# ---------------------------------------------------------------- model
def load_model():
    d = Path("/tmp/ecgf"); d.mkdir(exist_ok=True)
    for f in ("net1d.py", "tasks.txt"):
        if not (d / f).exists(): urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/{f}", d / f)
    sys.path.insert(0, str(d)); from net1d import Net1D
    labels = [l.strip() for l in open(d / "tasks.txt") if l.strip()]
    m = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
              kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=len(labels))
    ck = torch.load(hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"]); return m.to(DEV).eval(), labels


@torch.no_grad()
def forward_all(m, X):
    """-> probabilities (N,150), penultimate features (N,1024), stage-(n-2) activations for fine-tuning (fp16, CPU)."""
    P, F, H = [], [], []
    for i in range(0, len(X), 64):
        x = torch.from_numpy(np.stack(X[i:i + 64])).to(DEV)
        o = m.first_activation(m.first_conv(x))
        for s in m.stage_list[:-1]: o = s(o)
        H.append(o.half().cpu()); o2 = m.stage_list[-1](o); f = o2.mean(-1)
        P.append(torch.sigmoid(m.dense(f)).cpu().numpy()); F.append(f.cpu().numpy())
        if (i // 64) % 40 == 0: log(f"PROGRESS forward {i + len(x)}/{len(X)}")
    return np.concatenate(P), np.concatenate(F), torch.cat(H)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def train_head(Fe, y, tr, va, hidden=256, epochs=200, lr=1e-3, finetune=None, seed=0):
    """MLP head (dropout 0.2); early stopping on the validation fold by AUROC. finetune=(model, H) also trains the last stage."""
    torch.manual_seed(seed)
    din = Fe.shape[1]
    head = torch.nn.Sequential(torch.nn.Linear(din, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1)).to(DEV)
    params = [{"params": head.parameters(), "lr": lr}]
    if finetune:
        mdl, H = finetune; last = mdl.stage_list[-1]; [p.requires_grad_(True) for p in last.parameters()]
        params.append({"params": last.parameters(), "lr": 1e-5}); epochs = 5
    opt = torch.optim.Adam(params, weight_decay=1e-4); yt = torch.tensor(y, dtype=torch.float32)
    mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-6
    def feats(idx, train):
        if not finetune: return torch.from_numpy((Fe[idx] - mu) / sd).float().to(DEV)
        mdl.stage_list[-1].train(train)
        with torch.set_grad_enabled(train):
            f = mdl.stage_list[-1](H[idx].float().to(DEV)).mean(-1)
        return (f - torch.from_numpy(mu).to(DEV)) / torch.from_numpy(sd).to(DEV)
    def predict(idx):
        head.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 512): out.append(torch.sigmoid(head(feats(idx[i:i + 512], False))).squeeze(1).cpu().numpy())
        return np.concatenate(out)
    best, best_state, bad = -1, None, 0; tri = np.where(tr)[0]; vai = np.where(va)[0]; rng = np.random.default_rng(seed)
    for ep in range(epochs):
        head.train(); rng.shuffle(tri)
        for i in range(0, len(tri), 128):
            b = tri[i:i + 128]; loss = torch.nn.functional.binary_cross_entropy_with_logits(head(feats(b, True)).squeeze(1), yt[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        a = auroc(y[vai], predict(vai))
        if a > best: best, bad = a, 0; best_state = ({k: v.clone() for k, v in head.state_dict().items()},
                                                       {k: v.clone() for k, v in mdl.stage_list[-1].state_dict().items()} if finetune else None)
        else: bad += 1
        if bad >= (2 if finetune else 15): break
    head.load_state_dict(best_state[0])
    if finetune: mdl.stage_list[-1].load_state_dict(best_state[1])
    log(f"head trained: epochs {ep + 1}, best val AUROC {best:.4f}{' (finetune)' if finetune else ''}")
    return predict(np.arange(len(Fe))), best


def cluster_boot(flags, clusters, n=4000, seed=0):
    by = {}
    for f, c in zip(flags, clusters): by.setdefault(c, []).append(f)
    ks = list(by); rng = np.random.default_rng(seed); est = float(np.mean(flags))
    bs = [np.mean([f for k in rng.choice(ks, len(ks)) for f in by[k]]) for _ in range(n)]
    return [round(est, 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], len(ks)]


def paired_diff_boot(fa, fb, clusters, n=4000, seed=1):
    by = {}
    for a, b, c in zip(fa, fb, clusters): by.setdefault(c, []).append((a, b))
    ks = list(by); rng = np.random.default_rng(seed); est = float(np.mean(fa) - np.mean(fb)); bs = []
    for _ in range(n):
        s = [p for k in rng.choice(ks, len(ks)) for p in by[k]]; bs.append(np.mean([p[0] for p in s]) - np.mean([p[1] for p in s]))
    return [round(est, 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]]


def step3b(D, P, Fe, li):
    """F_STEP3B_PREREG_2026-11-16: M3 trained on folds 1-6, early-stopped on fold 7, thresholds on fold 8 only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ptb = D.source == "ptbxl"; lab_ok = D.group.isin(["sr_red", "normal", "green"]); yc = (D.group == "sr_red").astype(int).values
    tr = (ptb & D.fold.between(1, 6) & lab_ok).values; va = (ptb & (D.fold == 7) & lab_ok).values
    thr_mask = (ptb & (D.fold == 8) & (D.group == "sr_red")).values
    S = {"M0_model": P[:, li["ABNORMAL ECG"]]}
    for sd in (0, 1, 2):
        log(f"PHASE_START M3 seed {sd}"); S[f"M3_mlp_s{sd}"], _ = train_head(Fe, yc, tr, va, seed=sd)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)).fit(Fe[tr | va], yc[tr | va])
    S["linear_criteria"] = clf.predict_proba(Fe)[:, 1]
    test_ptb = ptb & D.fold.isin([9, 10]); ch = D.source == "chapman"
    nor_n = (D.source == "norwegian") & (D.group == "normal"); pf = D.source == "pf12red"; ath = nor_n | pf; cl = D.cluster[ath].tolist()
    rate = lambda s, t, m: round(float((s[m.values] > t).mean()), 4)
    res = {"n": {"fold8_sr_red": int(thr_mask.sum()), "test_ptb_sr_red": int((test_ptb & (D.group == "sr_red")).sum()), "athletes": int(ath.sum())}, "methods": {}}
    for name, s in S.items():
        res["methods"][name] = {}
        for tg in (0.90, 0.95):
            t = float(np.quantile(s[thr_mask], 1 - tg))
            res["methods"][name][str(tg)] = {"thr": round(t, 5), "sens_fold8": rate(s, t, pd.Series(thr_mask)),
                "sens_ptb_test": rate(s, t, test_ptb & (D.group == "sr_red")), "sens_chapman": rate(s, t, ch & (D.group == "sr_red")),
                "fa_green_ptb_test": rate(s, t, test_ptb & (D.group == "green")), "fa_normal_ptb_test": rate(s, t, test_ptb & (D.group == "normal")),
                "fa_green_chapman": rate(s, t, ch & (D.group == "green")), "fa_normal_chapman": rate(s, t, ch & (D.group == "normal")),
                "afa_norwegian": cluster_boot((s[nor_n.values] > t).astype(float), D.cluster[nor_n].tolist()),
                "afa_pf12red": cluster_boot((s[pf.values] > t).astype(float), D.cluster[pf].tolist()),
                "afa_pooled": cluster_boot((s[ath.values] > t).astype(float), cl)}
            r = res["methods"][name][str(tg)]
            log(f"METHOD {name} @{tg}: sens fold8 {r['sens_fold8']} ptb {r['sens_ptb_test']} ch {r['sens_chapman']} | AFA {r['afa_pooled']} | green ptb {r['fa_green_ptb_test']} normal ptb {r['fa_normal_ptb_test']}")
    m0 = res["methods"]["M0_model"]["0.9"]; t0 = m0["thr"]; f0 = (S["M0_model"][ath.values] > t0).astype(float)
    res["F3b_H1"] = {}
    for sd in (0, 1, 2):
        k = f"M3_mlp_s{sd}"; r = res["methods"][k]["0.9"]; f3 = (S[k][ath.values] > r["thr"]).astype(float); d = paired_diff_boot(f3, f0, cl)
        res["F3b_H1"][k] = {"afa_m3": r["afa_pooled"][0], "afa_m0": m0["afa_pooled"][0], "diff": d, "sens_m3": r["sens_ptb_test"], "sens_m0": m0["sens_ptb_test"],
                            "chapman_sens_m3": r["sens_chapman"],
                            "PASS": bool(r["afa_pooled"][0] <= 0.5 * m0["afa_pooled"][0] and d[1][1] < 0 and r["sens_ptb_test"] >= m0["sens_ptb_test"] - 0.02 and r["sens_chapman"] >= 0.80)}
    res["PRIMARY_PASS"] = res["F3b_H1"]["M3_mlp_s0"]["PASS"]; res["ROBUST_ALL_SEEDS"] = all(v["PASS"] for v in res["F3b_H1"].values())
    pd.concat([D.drop(columns=["path"]), pd.DataFrame(S)], axis=1).to_csv(OUT / "per_ecg_3b.csv", index=False)
    json.dump(res, open(OUT / "f3b.json", "w"), indent=1, default=str)
    log("F3B_H1 " + json.dumps(res["F3b_H1"]) + f" | PRIMARY_PASS {res['PRIMARY_PASS']} ROBUST {res['ROBUST_ALL_SEEDS']}")
    print("F3B_RESULT " + json.dumps(res, default=str), flush=True); log("F3B_DONE")


def main():
    log(f"PHASE_START cohorts (smoke={SMOKE}, nproc={NPROC})")
    items = ptbxl_items() + chapman_items(); nor, sl12 = norwegian_items(); items += nor
    log(f"items {len(items)}; PHASE_START load+measure")
    X, M, meta = [], [], []
    with Pool(NPROC) as pool:
        for k, r in enumerate(pool.imap(load_item, items, chunksize=8)):
            if r is not None: X.append(r[0]); M.append(r[1]); meta.append(r[2])
            if (k + 1) % 1000 == 0:
                el = time.time() - T0; log(f"PROGRESS load {k + 1}/{len(items)} elapsed {el / 60:.1f} min")
    pX, pM, pmeta, stmts = pf12red(); X += pX; M += pM; meta += pmeta
    D = pd.DataFrame(meta, columns=["source", "group", "path", "fold", "sex", "cluster", "cohort"])
    D = pd.concat([D, pd.DataFrame(M)], axis=1)
    D["qtc"] = D.qt / np.sqrt(60 / D.hr)
    log("loaded: " + json.dumps({f"{a}/{b}": int(v) for (a, b), v in D.groupby(["source", "group"]).size().items()}))
    log("measurement success (all four finite): " + json.dumps(D.groupby("source").apply(lambda g: round(float(np.isfinite(g[["hr", "pr", "qrs", "qt"]]).all(1).mean()), 3)).to_dict()))
    log("PHASE_START forward")
    model, labels = load_model(); P, Fe, H = forward_all(model, X)
    li = {l: i for i, l in enumerate(labels)}; miss = [c for c in SR_MORPH + SR_WIDTH_QT + list(BORDER) if c not in li]
    log(f"concepts missing from tasks.txt: {miss}")
    if os.environ.get("F3B") == "1":
        return step3b(D, P, Fe, li)
    # ------------------------------------------------ scores
    tune = (D.source == "ptbxl") & D.fold.between(1, 8); neg_t = tune & D.group.isin(["normal", "green"])
    def ecdf_cols(names):
        idx = [li[c] for c in names if c in li]; ref = np.sort(P[neg_t.values][:, idx], 0)
        return np.stack([np.searchsorted(ref[:, j], P[:, i], side="right") / len(ref) for j, i in enumerate(idx)], 1)
    Zm1 = ecdf_cols(SR_MORPH + SR_WIDTH_QT); Zm2 = ecdf_cols(SR_MORPH); Zb = ecdf_cols(sorted(BORDER))
    b2 = np.sort(Zb, 1)[:, -2]
    meas_flag = ((D.hr < 30) | (D.pr >= 400) | (D.qrs >= 140) | (D.qtc >= np.where(D.sex == "F", 480, 470)) | (D.qtc <= 320)).fillna(False).values
    S = {"M0_model": P[:, li["ABNORMAL ECG"]]}
    iR = [li[l] for l in labels if l not in GREEN | BORDER | IGNORE]; iB = [li[l] for l in BORDER if l in li]
    S["Fix2_gate100"] = np.maximum(P[:, iR].max(1), np.sort(P[:, iB], 1)[:, -2])
    S["M1_screen_gate"] = np.maximum(Zm1.max(1), b2)
    S["M2_measure_then_ask"] = np.where(meas_flag, 2.0, np.maximum(Zm2.max(1), b2))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr_lin = tune & D.group.isin(["sr_red", "other_red", "normal", "green"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)).fit(Fe[tr_lin.values], D.group[tr_lin].isin(["sr_red", "other_red"]).astype(int))
    S["Fix1_linear"] = clf.predict_proba(Fe)[:, 1]
    yc = (D.group == "sr_red").astype(int).values; lab_ok = D.group.isin(["sr_red", "normal", "green"])
    tr = ((D.source == "ptbxl") & D.fold.between(1, 7) & lab_ok).values; va = ((D.source == "ptbxl") & (D.fold == 8) & lab_ok).values
    log("PHASE_START M3 MLP"); S["M3_mlp"], m3_auc = train_head(Fe, yc, tr, va)
    try:
        log("PHASE_START M3b finetune"); S["M3b_mlp_finetune"], m3b_auc = train_head(Fe, yc, tr, va, finetune=(model, H))
    except Exception as e:
        log(f"M3b failed: {type(e).__name__}: {str(e)[:200]}"); m3b_auc = None
    pd.concat([D.drop(columns=["path"]), pd.DataFrame(S)], axis=1).to_csv(OUT / "per_ecg.csv", index=False)
    # ------------------------------------------------ operating curves
    test_ptb = (D.source == "ptbxl") & D.fold.isin([9, 10]); ch = D.source == "chapman"
    nor_n = (D.source == "norwegian") & (D.group == "normal"); pf = D.source == "pf12red"; ath = nor_n | pf
    srt = (tune & (D.group == "sr_red")).values
    res = {"n": {k: int(v) for k, v in {"tune_sr_red": srt.sum(), "test_ptb_sr_red": (test_ptb & (D.group == "sr_red")).sum(),
           "chapman_sr_red": (ch & (D.group == "sr_red")).sum(), "norwegian_normal": nor_n.sum(), "pf12red_recordings": pf.sum(),
           "pf12red_players": D.cluster[pf].nunique()}.items()}, "methods": {}, "m3_val_auroc": m3_auc, "m3b_val_auroc": m3b_auc,
           "M4_ecgfm": "not run in this job"}
    rate = lambda s, t, mask: round(float((s[mask.values] > t).mean()), 4) if mask.sum() else None
    for name, s in S.items():
        rows = {}
        for tg in TARGETS:
            t = float(np.quantile(s[srt], 1 - tg))
            rows[str(tg)] = {"thr": round(t, 5), "sens_ptb_test": rate(s, t, test_ptb & (D.group == "sr_red")), "sens_chapman": rate(s, t, ch & (D.group == "sr_red")),
                             "fa_green_ptb_test": rate(s, t, test_ptb & (D.group == "green")), "fa_normal_ptb_test": rate(s, t, test_ptb & (D.group == "normal")),
                             "fa_green_chapman": rate(s, t, ch & (D.group == "green")), "fa_normal_chapman": rate(s, t, ch & (D.group == "normal")),
                             "other_red_flagged_ptb_test": rate(s, t, test_ptb & (D.group == "other_red")),
                             "afa_norwegian": cluster_boot((s[nor_n.values] > t).astype(float), D.cluster[nor_n].tolist()),
                             "afa_pf12red": cluster_boot((s[pf.values] > t).astype(float), D.cluster[pf].tolist()),
                             "afa_pooled": cluster_boot((s[ath.values] > t).astype(float), D.cluster[ath].tolist())}
            rows[str(tg)]["referrals_per_1000"] = [round(1000 * v, 1) if isinstance(v, float) else [round(1000 * w, 1) for w in v] for v in rows[str(tg)]["afa_pooled"][:2]]
        res["methods"][name] = rows
        r9 = rows["0.9"]; log(f"METHOD {name} @0.90: sens ptb {r9['sens_ptb_test']} ch {r9['sens_chapman']} | AFA pooled {r9['afa_pooled']} | green ptb {r9['fa_green_ptb_test']} normal ptb {r9['fa_normal_ptb_test']}")
    # ------------------------------------------------ hypotheses
    t0 = float(np.quantile(S["M0_model"][srt], 0.10)); t2 = float(np.quantile(S["M2_measure_then_ask"][srt], 0.10)); t3 = float(np.quantile(S["M3_mlp"][srt], 0.10))
    f0 = (S["M0_model"][ath.values] > t0).astype(float); f2 = (S["M2_measure_then_ask"][ath.values] > t2).astype(float); f3 = (S["M3_mlp"][ath.values] > t3).astype(float)
    cl = D.cluster[ath].tolist(); m0, m2, m3 = res["methods"]["M0_model"], res["methods"]["M2_measure_then_ask"], res["methods"]["M3_mlp"]
    d20 = paired_diff_boot(f2, f0, cl); d30 = paired_diff_boot(f3, f0, cl)
    res["F3_H1"] = {"afa_m2": m2["0.9"]["afa_pooled"][0], "afa_m0": m0["0.9"]["afa_pooled"][0], "diff_m2_minus_m0": d20,
                    "sens_m2": m2["0.9"]["sens_ptb_test"], "sens_m0": m0["0.9"]["sens_ptb_test"],
                    "PASS": bool(m2["0.9"]["afa_pooled"][0] <= 0.5 * m0["0.9"]["afa_pooled"][0] and d20[1][1] < 0 and m2["0.9"]["sens_ptb_test"] >= m0["0.9"]["sens_ptb_test"] - 0.02)}
    res["F3_H2"] = {"PASS": all(m2[str(t)]["afa_pooled"][0] < m0[str(t)]["afa_pooled"][0] for t in TARGETS)}
    res["F3_H3"] = {"afa_m3": m3["0.9"]["afa_pooled"][0], "diff_m3_minus_m0": d30, "sens_m3": m3["0.9"]["sens_ptb_test"],
                    "PASS": bool(m3["0.9"]["afa_pooled"][0] <= 0.5 * m0["0.9"]["afa_pooled"][0] and d30[1][1] < 0 and m3["0.9"]["sens_ptb_test"] >= m0["0.9"]["sens_ptb_test"] - 0.02)}
    res["transfer_guard_chapman_sens_ge_0.80_at_0.90"] = {k: (v["0.9"]["sens_chapman"] or 0) >= 0.80 for k, v in res["methods"].items()}
    # Pareto at each target over pooled AFA (lower better) and realised test sensitivity (higher better)
    res["pareto"] = {}
    for tg in TARGETS:
        pts = {k: (v[str(tg)]["sens_ptb_test"], v[str(tg)]["afa_pooled"][0]) for k, v in res["methods"].items()}
        res["pareto"][str(tg)] = sorted(k for k, (s, a) in pts.items() if not any(s2 >= s and a2 <= a and (s2 > s or a2 < a) for k2, (s2, a2) in pts.items() if k2 != k))
    # ------------------------------------------------ device algorithm (exploratory)
    nn = D[nor_n]; res["M5_device_GE12SL"] = {"norwegian_normal_flag_rate": round(float(np.mean([sl12.get(c, False) for c in nn.cluster])), 4)}
    st_all = [s for v in stmts.values() for s in v]
    res["M5_device_GE12SL"]["pf12red_statement_elements_found"] = len(st_all)
    if st_all:
        fl = [any(re.search(r"borderline ecg|abnormal ecg", s, re.I) for s in v) for v in stmts.values()]
        res["M5_device_GE12SL"]["pf12red_flag_rate"] = round(float(np.mean(fl)), 4)
        res["M5_device_GE12SL"]["pf12red_top_statements"] = pd.Series(st_all).str[:60].value_counts().head(15).to_dict()
    # ------------------------------------------------ mechanism (secondary)
    mech = {}
    for src in ("ptbxl", "chapman"):
        g = D[(D.source == src) & D.group.isin(["normal", "green"]) & np.isfinite(D.hr)]
        y = (S["M0_model"][g.index.values] > t0).astype(int); Xm = np.c_[(g.hr.values - 60) / 10, (g.group == "green").astype(float).values]
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xm, y)
        mech[src] = {"n": int(len(g)), "or_per_10bpm_lower": round(float(np.exp(-lr.coef_[0][0])), 3), "or_green": round(float(np.exp(lr.coef_[0][1])), 3),
                     "hr_median_green": round(float(g[g.group == "green"].hr.median()), 1), "hr_median_normal": round(float(g[g.group == "normal"].hr.median()), 1),
                     "flag_rate_by_hr": {b: round(float(y[(g.hr.values >= lo) & (g.hr.values < hi)].mean()), 3) if ((g.hr.values >= lo) & (g.hr.values < hi)).any() else None
                                         for b, (lo, hi) in {"<50": (0, 50), "50-59": (50, 60), "60-79": (60, 80), ">=80": (80, 400)}.items()}}
    res["mechanism_M0_at_0.90"] = mech
    res["measurements_athletes"] = {c: {k: round(float(D[D.source == c][k].median()), 1) for k in ("hr", "pr", "qrs", "qtc")} for c in ("norwegian", "pf12red")}
    res["m2_measurement_flags"] = {c: int(meas_flag[(D.source == c).values].sum()) for c in ("ptbxl", "chapman", "norwegian", "pf12red")}
    json.dump(res, open(OUT / "f3.json", "w"), indent=1, default=str)
    log("F3_H1 " + json.dumps(res["F3_H1"]) + " | F3_H2 " + json.dumps(res["F3_H2"]) + " | F3_H3 " + json.dumps(res["F3_H3"]))
    print("F3_RESULT " + json.dumps(res, default=str), flush=True)
    log("F3_DONE")


if __name__ == "__main__":
    main()
