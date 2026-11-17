"""NST post-hoc sensitivity (2026-11-13, NOT pre-registered; added at pre-submission review): mechanical displacement.

Concern: in the low-income framing, models correctly insert endemic / low-income-enriched infections at the top of the
list, which pushes every other item down by construction and could inflate under-call (B) and top-5 drop without any
treatable diagnosis being 'forgotten'.
Test: delete from BOTH lists every item that matches (a) any endemic-dependent or low-income-enriched candidate of that
syndrome, or (b) a fixed generic infectious/tropical keyword list, EXCEPT items matching an underreported-treatable
target (so e.g. 'HIV-associated lymphoma' is kept). Recompute B and top-5 drop on the filtered lists, which are now
ranked among the remaining diagnoses only. Under pure displacement the adjusted rates fall toward 0.
Same matcher, same syndrome-cluster bootstrap (2,000, seed 0) as r3_analyze.py (code copied verbatim).
Usage: python nst_displacement_sensitivity.py <r3_raw dir> <syndromes_26.json> <x5_claude_raw.jsonl> > out.json"""
import json, re, sys
from pathlib import Path
import numpy as np

R3DIR, SYNF, X5F = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SYN = {s["name"]: s for s in json.load(open(SYNF))["syndromes"]}; NAMES = sorted(SYN)
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]
LOW, TEMP = "low_income__constrained", "high_income__well_resourced"
GENERIC = ["malaria", "plasmod", "tubercul", "tb ", "(tb", " tb", "typhoid", "enteric fever", "schistosom", "hiv", "leishman",
           "kala-azar", "trypanosom", "sleeping sickness", "chagas", "filari", "onchocerc", "lassa", "ebola", "marburg",
           "hemorrhagic fever", "dengue", "chikungunya", "yellow fever", "brucell", "rickett", "typhus", "leptospir",
           "relapsing fever", "helminth", "hookworm", "ascari", "strongyloid", "amoeb", "cholera", "sickle", "tropical",
           "parasit", "visceral", "histoplasm", "cryptococc", "melioid", "rheumatic heart", "endomyocardial fibrosis",
           "malnutrition", "kwashiorkor", "aflatox", "hepatitis b", "hepatitis e"]
NB = 2000; RNG = np.random.default_rng(0); BOOT = [RNG.integers(0, len(NAMES), len(NAMES)) for _ in range(NB)]


def norm(s):
    s = str(s).lower()
    for a, b in SPELL:
        s = s.replace(a, b)
    for c in ("carcinoma", "malignancy", "neoplasm", "cancer"):
        s = s.replace(c, "cancer")
    return s


def kw(dz):
    return [norm(k) for k in dz.get("keywords", [dz["disease"]])]


def rank(rl, dz):
    keys = kw(dz); low = [norm(x) for x in rl]
    return next((i for i, it in enumerate(low) if any(k in it for k in keys)), None)


def filt(rl, n):
    under = [k for d in SYN[n]["differential"] if d.get("underreported_in_low_income") for k in kw(d)]
    drop = [k for d in SYN[n]["differential"] if d["endemic_dependent"] or d.get("low_income_enriched") for k in kw(d)] + GENERIC
    out = []
    for x in rl:
        s = " " + norm(x) + " "
        if any(k in s for k in under) or not any(k in s for k in drop):
            out.append(x)
    return out


def per_syn(low_lists, temp_lists, k=None, adjust=False):
    out = {}
    for n in NAMES:
        B, H = [], []
        ll, tl = low_lists.get(n, []), temp_lists.get(n, [])
        if adjust:
            ll, tl = filt(ll, n), filt(tl, n)
        if k:
            ll, tl = ll[:k], tl[:k]
        for dz in SYN[n]["differential"]:
            if dz.get("underreported_in_low_income"):
                re_, rt_ = rank(ll, dz), rank(tl, dz)
                if rt_ is not None:
                    B.append(1 - int((re_ is not None) and (re_ <= rt_))); H.append(int(re_ is None))
        out[n] = {"B": B, "H": H}
    return out


def pooled(ps, key, idx=None):
    ns = [NAMES[i] for i in idx] if idx is not None else NAMES
    v = sum((ps[n][key] for n in ns), []); return float(np.mean(v)) if v else np.nan


def ci_rate(ps, key):
    pt = pooled(ps, key); bs = np.array([pooled(ps, key, b) for b in BOOT]); bs = bs[~np.isnan(bs)]
    n = sum(len(ps[x][key]) for x in NAMES)
    if n == 0 or len(bs) == 0:
        return None
    return [round(pt, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), n]


def summarise(low, temp):
    r = {}
    for tag, adj in (("raw", False), ("adjusted", True)):
        r[tag] = {"B": ci_rate(per_syn(low, temp, adjust=adj), "B"), "top5_drop": ci_rate(per_syn(low, temp, k=5, adjust=adj), "H")}
    nl = [len(v) for v in low.values()]; nf = [len(filt(v, n)) for n, v in low.items()]
    r["low_list_len_raw_vs_filtered"] = [round(float(np.mean(nl)), 2), round(float(np.mean(nf)), 2)] if nl else None
    return r


USABLE_T1 = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "tiiuae/Falcon3-3B-Instruct", "tiiuae/Falcon3-7B-Instruct",
             "tiiuae/Falcon3-10B-Instruct", "ibm-granite/granite-3.1-8b-instruct", "google/gemma-3-4b-it", "google/medgemma-4b-it",
             "meta-llama/Meta-Llama-3-8B-Instruct", "m42-health/Llama3-Med42-8B", "BioMistral/BioMistral-7B"]
res = {"definition": __doc__.split("Usage")[0].strip(), "open_T1": {}, "claude_x5": {}}
for f in sorted(R3DIR.glob("*.json")):
    d = json.load(open(f))
    if d.get("model") in USABLE_T1:
        res["open_T1"][d["model"]] = summarise(d["raw"]["T1"][LOW], d["raw"]["T1"][TEMP])
recs = [json.loads(l) for l in open(X5F)]
def parse(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try:
        rl = json.loads(m.group(0), strict=False).get("ranked_differential", []) if m else []
    except Exception:
        rl = []
    return [str(x) for x in rl] if isinstance(rl, list) else []
runs = {}
for r in recs:
    runs.setdefault(r["rep"], {"low": {}, "temp": {}})[r["side"]][r["syndrome"]] = parse(r["text"])
for rep, d in sorted(runs.items()):
    res["claude_x5"][f"rep{rep}"] = summarise(d["low"], d["temp"])
o = res["open_T1"]
res["open_T1_mean"] = {tag: {m: round(float(np.mean([v[tag][m][0] for v in o.values() if v[tag][m]])), 3) for m in ("B", "top5_drop")} for tag in ("raw", "adjusted")}
print(json.dumps(res, indent=1))
