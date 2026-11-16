"""Programme Idea F step 2 (F_STEP2_PREREG_2026-11-10.md): International-Criteria gate over ECGFounder's named concepts. Derived from step 1b (PF12RED, GE CardioSoft XML); derived from step 1 (F_STEP1_PREREG_2026-11-10.md): ECGFounder over-call on REAL athletes.
Pipeline identical to CV_Research/athlete_ecg/p1_overcall_job.py. PTB-XL (Kaggle input) for the control-calibrated
threshold + P1 sanity replication; Norwegian athlete DB (PhysioNet, CC BY 4.0) and PF12RED (GitHub, CC0) downloaded at
runtime. Prints label structures verbatim, then F1_RESULT lines; writes /kaggle/working/f1.json. Private kernel."""
import ast, base64, glob, io, json, os, re, subprocess, sys, time, urllib.request, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wfdb", "openpyxl"], check=False)
from pathlib import Path
import numpy as np, pandas as pd, torch, wfdb
from huggingface_hub import hf_hub_download

T0 = time.time(); DEV = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda m: print(f"[f1 {time.time() - T0:6.0f}s] {m}", flush=True)       # noqa: E731
GH = "https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/{}"
GREEN_PTB = {"SBRAD", "SARRH", "1AVB", "IRBBB", "VCLVH", "HVOLT"}
W = Path("/kaggle/working"); TMP = Path("/tmp/f1"); TMP.mkdir(exist_ok=True)


def fetch_code():
    d = Path("/tmp/ecgf"); d.mkdir(exist_ok=True)
    for f in ("net1d.py", "tasks.txt"):
        if not (d / f).exists():
            urllib.request.urlretrieve(GH.format(f), d / f)
    sys.path.insert(0, str(d))
    return [l.strip() for l in open(d / "tasks.txt") if l.strip()]


def load_model(n):
    from net1d import Net1D
    m = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
              use_bn=False, use_do=False, n_classes=n)
    ck = torch.load(hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False)
    return m.to(DEV).eval()


def filter_bandpass(signal, fs):
    """Verbatim from ECGFounder util.py (MIT)."""
    from scipy.signal import butter, filtfilt, iirnotch, medfilt
    b, a = iirnotch(50, 30, fs); f = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        f[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        f[c] = filtfilt(b, a, f[c])
    k = int(0.4 * fs) + 1; k += (k % 2 == 0)
    from scipy.signal import medfilt as mf
    return f - np.stack([mf(f[c], kernel_size=k) for c in range(f.shape[0])])


def prep(sig, fs=500):
    """sig: (T, 12) in standard lead order I II III aVR aVL aVF V1..V6."""
    x = np.nan_to_num(sig.T.astype(np.float64))
    n10 = int(10 * fs); x = x[:, :n10] if x.shape[1] >= n10 else np.pad(x, ((0, 0), (0, n10 - x.shape[1])))
    if x.shape[1] != 5000:
        x = np.stack([np.interp(np.linspace(0, x.shape[1] - 1, 5000), np.arange(x.shape[1]), c) for c in x])
    x = filter_bandpass(x, 500)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def reorder(sig, names):
    names = [n.upper().replace("-", "") for n in names]; idx = {n: i for i, n in enumerate(names)}
    if all(l in idx for l in LEADS):
        return sig[:, [idx[l] for l in LEADS]]
    if all(l in idx for l in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")):      # derive limb leads
        I, II = sig[:, idx["I"]], sig[:, idx["II"]]; III = II - I
        return np.stack([I, II, III, -(I + II) / 2, I - II / 2, II - I / 2] + [sig[:, idx[f"V{k}"]] for k in range(1, 7)], 1)
    raise ValueError(f"leads {names}")


# ---------------------------------------------------------------- cohorts
def ptbxl_items():
    csv = next(iter(glob.glob("/kaggle/input/**/ptbxl_database.csv", recursive=True)))
    root = Path(csv).parent; d = pd.read_csv(csv)
    codes = d.scp_codes.apply(lambda s: {k for k, v in ast.literal_eval(s).items() if v >= 50 or k in GREEN_PTB | {"SR"}})
    green = codes.apply(lambda c: bool(c) and c <= GREEN_PTB | {"SR", "NORM"} and bool(c & GREEN_PTB))
    norm = codes.apply(lambda c: c <= {"SR", "NORM"} and "NORM" in c)
    g = d[green].sample(min(1500, green.sum()), random_state=0); n = d[norm].sample(2000, random_state=0)
    return [("ptbxl", "green", str(root / r.filename_hr)) for r in g.itertuples()] + \
           [("ptbxl", "control", str(root / r.filename_hr)) for r in n.itertuples()]


def norwegian():
    base = "https://physionet.org/files/norwegian-athlete-ecg/1.0.0/"
    recs = urllib.request.urlopen(base + "RECORDS", timeout=60).read().decode().split()
    out = []
    for r in recs:
        for ext in (".hea", ".dat"):
            p = TMP / f"{r}{ext}"
            if not p.exists():
                urllib.request.urlretrieve(base + r + ext, p)
        h = wfdb.rdheader(str(TMP / r)); out.append((r, h.comments))
    log("NORWEGIAN header comments (first 3, verbatim): " + json.dumps([c for _, c in out[:3]]))
    return out


def pf12red():
    z = TMP / "pf.zip"
    if not z.exists():
        urllib.request.urlretrieve("https://codeload.github.com/dradolfomunoz/PF12RED/zip/refs/heads/main", z)
    zf = zipfile.ZipFile(z); names = zf.namelist()
    lab = next((n for n in names if n.endswith("LabelData.xlsx")), None)
    labels = pd.read_excel(io.BytesIO(zf.read(lab))) if lab else None
    if labels is not None:
        log(f"PF12RED LabelData columns {list(labels.columns)}; head {labels.head(3).to_dict('records')}")
        for c in labels.columns:
            vc = labels[c].astype(str).value_counts()
            if 1 < len(vc) <= 25:
                log(f"  {c}: {vc.to_dict()}")
    xmls = sorted(n for n in names if n.lower().endswith(".xml"))
    if xmls:
        head = zf.read(xmls[0])[:1500].decode(errors="ignore"); log("PF12RED first XML head: " + re.sub(r"\s+", " ", head)[:1200])
    return zf, xmls, labels


def parse_muse(xml_bytes):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes); leads = {}; fs = 500
    for wf in root.iter("Waveform"):
        typ = (wf.findtext("WaveformType") or "").lower()
        if "rhythm" not in typ:
            continue
        fs = int(wf.findtext("SampleBase") or 500)
        for ld in wf.iter("LeadData"):
            lid = ld.findtext("LeadID"); data = ld.findtext("WaveFormData")
            gain = float(ld.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
            if lid and data:
                leads[lid] = np.frombuffer(base64.b64decode(data), dtype="<i2").astype(np.float64) * gain / 1000.0
    if not leads:
        raise ValueError("no MUSE rhythm leads")
    L = min(len(v) for v in leads.values()); names = list(leads)
    return np.stack([leads[n][:L] for n in names], 1), names, fs


@torch.no_grad()
def score(m, sigs):
    out = []
    for i in range(0, len(sigs), 64):
        x = torch.tensor(np.stack(sigs[i:i + 64])).to(DEV); out.append(torch.sigmoid(m(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 150))


def wilson(k, n):
    if n == 0: return [None, None]
    z = 1.96; p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]



def parse_cardiosoft(xml_bytes):
    """GE CardioSoft <CardiologyXML>: StripData (10 s rhythm) with <WaveformData lead="I">comma ints</WaveformData>."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes); strip = root.find(".//StripData")
    if strip is None:
        raise ValueError("no StripData")
    fs = float((strip.findtext("SampleRate") or "500").strip()); res = float((strip.findtext("Resolution") or "1").strip())
    leads = {}
    for wd in strip.iter("WaveformData"):
        lid = wd.get("lead"); txt = (wd.text or "").strip()
        if lid and txt:
            leads[lid] = np.array([float(v) for v in re.split(r"[,\s]+", txt) if v], np.float64) * res / 1000.0   # uV -> mV
    if len(leads) < 8:
        raise ValueError(f"leads {list(leads)}")
    L = min(len(v) for v in leads.values()); names = list(leads)
    return np.stack([leads[n][:L] for n in names], 1), names, int(fs)



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
labels_ = fetch_code(); model = load_model(len(labels_)); IA = labels_.index("ABNORMAL ECG")
RED = [l for l in labels_ if l not in GREEN | BORDER | IGNORE]
iR = [labels_.index(l) for l in RED]; iB = [labels_.index(l) for l in BORDER if l in labels_]
log(f"concepts: green {len(GREEN)}, borderline {len(iB)}, red {len(iR)}, ignored {len(IGNORE)}")

def gate_score(P):
    """Monotone score for the gate rule: abnormal iff max(red) > t or (2nd-largest borderline) > t."""
    red = P[:, iR].max(1); b2 = np.sort(P[:, iB], 1)[:, -2]
    return np.maximum(red, b2)

def ptbxl_folds():
    csv = next(iter(glob.glob("/kaggle/input/**/ptbxl_database.csv", recursive=True))); root = Path(csv).parent; d = pd.read_csv(csv)
    codes = d.scp_codes.apply(lambda s: {k for k, v in ast.literal_eval(s).items() if v >= 50 or k in GREEN_PTB | {"SR"}})
    grp = codes.apply(lambda c: "red" if c - GREEN_PTB - {"SR", "NORM"} else ("green" if c & GREEN_PTB else ("normal" if "NORM" in c else "skip")))
    d = d.assign(group=grp); d = d[d.group != "skip"]
    pick = pd.concat([g.sample(min(len(g), n), random_state=0) for (gname, g), n in
                      zip(d.groupby("group"), [2500, 2500, 2500])])
    return [(r.group, int(r.strat_fold), str(root / r.filename_hr)) for r in pick.itertuples()]

items = ptbxl_folds(); sig, meta = [], []
for g, f, p in items:
    try:
        s_, fd = wfdb.rdsamp(p); sig.append(prep(s_, fd["fs"])); meta.append((g, f))
    except Exception:
        pass
P = score(model, sig); G = np.array([m[0] for m in meta]); F = np.array([m[1] for m in meta])
gs, ab = gate_score(P), P[:, IA]
cal, tst = F <= 8, F >= 9
t_gate = float(np.quantile(gs[cal & (G == "red")], 0.10)); t_ab = float(np.quantile(ab[cal & (G == "red")], 0.10))
def rates(score, t, mask):
    return {g: round(float((score[mask & (G == g)] > t).mean()), 3) for g in ("red", "green", "normal")}
res = {"thresholds": {"gate": round(t_gate, 4), "abnormal_head": round(t_ab, 4)},
       "ptbxl_test": {"gate": rates(gs, t_gate, tst), "abnormal_head": rates(ab, t_ab, tst)}, "n_test": int(tst.sum())}
pt = res["ptbxl_test"]
res["F2_H1"] = bool(pt["gate"]["green"] <= 0.5 * pt["abnormal_head"]["green"] and pt["gate"]["normal"] - pt["abnormal_head"]["normal"] <= 0.02)
log(f"PTB-XL test: {res['ptbxl_test']}  H1={res['F2_H1']}")
# athletes
nor = norwegian(); ns, ncat = [], []
for r, comments in nor:
    s_, fd = wfdb.rdsamp(str(TMP / r)); ns.append(prep(reorder(s_, fd["sig_name"]), fd["fs"]))
    ncat.append("normal" if "Normal ECG" in " ".join(c for c in comments if c.strip().startswith("C:")) else "other")
NP = score(model, ns); nm = np.array(ncat) == "normal"
res["norwegian_normal"] = {"n": int(nm.sum()), "gate": round(float((gate_score(NP)[nm] > t_gate).mean()), 3),
                           "abnormal_head": round(float((NP[nm, IA] > t_ab).mean()), 3)}
zf, xmls, lab = pf12red()
hdr = lab.iloc[0].tolist(); cols = [str(h) if str(h) != "nan" else str(c) for h, c in zip(hdr, lab.columns)]
L = lab.iloc[1:].copy(); L.columns = cols; L = L[pd.to_numeric(L[cols[0]], errors="coerce").notna()]
TW_BAD = ["I", "II", "aVL", "aVF", "V3", "V4", "V5", "V6"]
green_players = {str(int(float(r[cols[0]]))) for _, r in L.iterrows() if not any(str(r.get(ld, "")).strip().upper() == "X" for ld in TW_BAD if ld in L.columns)}
race = {str(int(float(r[cols[0]]))): str(r.get("Race", "")) for _, r in L.iterrows()}
ps, pl = [], []
for n in xmls:
    pid = n.split("/")[-1].split("_")[0]
    if pid not in green_players:
        continue
    try:
        s_, names, fs = parse_cardiosoft(zf.read(n)); ps.append(prep(reorder(s_, names), fs)); pl.append(pid)
    except Exception:
        pass
PP = score(model, ps)
def pboot(flags, players, n=4000):
    by = {}
    for f_, p_ in zip(flags, players): by.setdefault(p_, []).append(f_)
    ks = list(by); rng = np.random.default_rng(0); est = float(np.mean(flags))
    bs = [np.mean([f for k in rng.choice(ks, len(ks)) for f in by[k]]) for _ in range(n)]
    return [round(est, 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], len(ks)]
gflag, aflag = gate_score(PP) > t_gate, PP[:, IA] > t_ab
res["pf12red_green_only"] = {"gate": pboot(gflag, pl), "abnormal_head": pboot(aflag, pl),
                             "gate_by_race": {rc: round(float(np.mean([f for f, p in zip(gflag, pl) if race.get(p) == rc])), 3) for rc in sorted(set(race.get(p) for p in pl))}}
nn_, pf = res["norwegian_normal"], res["pf12red_green_only"]
res["F2_H2"] = bool(nn_["gate"] <= 0.5 * nn_["abnormal_head"] and pf["gate"][0] <= 0.5 * pf["abnormal_head"][0])
json.dump(res, open(W / "f2.json", "w"), default=str)
print("F2_RESULT " + json.dumps(res, default=str), flush=True)
