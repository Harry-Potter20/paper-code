"""Programme Idea F step 1b (PF12RED, GE CardioSoft XML); derived from step 1 (F_STEP1_PREREG_2026-11-10.md): ECGFounder over-call on REAL athletes.
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


labels_ = fetch_code(); model = load_model(len(labels_)); IA = labels_.index("ABNORMAL ECG")
def top_labels(Prow, k=3):
    idx = np.argsort(-Prow)[:k + 3]
    return [labels_[i] for i in idx if labels_[i] not in ("ABNORMAL ECG", "NORMAL ECG", "BORDERLINE ECG")][:k]


THR = 0.9845   # F step 1: PTB-XL NORM+SR 90th percentile (F_STEP1_PREREG RESULT); fixed, not recomputed
zf, xmls, lab = pf12red()
hdr = lab.iloc[0].tolist(); cols = [str(h) if str(h) != "nan" else str(c) for h, c in zip(hdr, lab.columns)]
L = lab.iloc[1:].copy(); L.columns = cols
L = L[pd.to_numeric(L[cols[0]], errors="coerce").notna()]
pid_col = cols[0]
TW_BAD = ["I", "II", "aVL", "aVF", "V3", "V4", "V5", "V6"]
def mark(v): return str(v).strip().upper() == "X"
meta = {}
for _, r in L.iterrows():
    pid = str(int(float(r[pid_col])))
    bad = [ld for ld in TW_BAD if ld in L.columns and mark(r[ld])]
    meta[pid] = {"race": str(r.get("Race", "")), "SB": mark(r.get("SB", "")), "iRBBB": mark(r.get("iRBBB", "")),
                 "twi_bad_leads": bad, "green_only": len(bad) == 0}
log(f"labels parsed for {len(meta)} players; green-only {sum(m['green_only'] for m in meta.values())}")
sigs, info = [], []
for n in xmls:
    fn = n.split("/")[-1]; pid = fn.split("_")[0]
    try:
        s_, names, fs = parse_cardiosoft(zf.read(n)); sigs.append(prep(reorder(s_, names), fs)); info.append({"file": fn, "player": pid, **meta.get(pid, {})})
    except Exception as e:
        info.append({"file": fn, "player": pid, "error": str(e)[:100]})
ok = [i for i in info if "error" not in i]
P = score(model, sigs); pa = P[:, IA] if len(sigs) else np.array([])
for i, p, row in zip(ok, pa, P):
    i["p_abn"] = round(float(p), 3); i["flag"] = bool(p > THR); i["top"] = top_labels(row)
log(f"parsed {len(ok)}/{len(xmls)}; errors {[i for i in info if 'error' in i][:3]}")
def player_boot(rows, n=4000, seed=0):
    by = {}
    for r in rows: by.setdefault(r["player"], []).append(r["flag"])
    ps = list(by); rng = np.random.default_rng(seed)
    if not ps: return None, [None, None], 0
    est = np.mean([f for p in ps for f in by[p]])
    bs = [np.mean([f for p in rng.choice(ps, len(ps)) for f in by[p]]) for _ in range(n)]
    return round(float(est), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], len(ps)
prim = [i for i in ok if i.get("green_only")]
res = {"threshold": THR, "n_xml": len(xmls), "n_parsed": len(ok),
       "overcall_all": player_boot(ok), "overcall_green_only_primary": player_boot(prim),
       "by_race_green_only": {rc: player_boot([i for i in prim if i.get("race") == rc]) for rc in sorted({i.get("race") for i in prim})},
       "top_labels_flagged_primary": __import__("collections").Counter(t for i in prim if i["flag"] for t in i["top"]).most_common(10),
       "per_record": ok}
json.dump(res, open(W / "f1b.json", "w"), default=str)
print("F1B_RESULT " + json.dumps({k: v for k, v in res.items() if k != "per_record"}, default=str), flush=True)
print("F1B_DONE", flush=True)
