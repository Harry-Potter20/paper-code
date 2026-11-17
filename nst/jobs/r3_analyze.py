"""NST R3 offline analysis (NST_R3_ROBUSTNESS_PREREG_2026-11-10.md). Reads the raw R3 model JSONs (+ optional X5 Claude
jsonl), scores with the paper's deterministic matcher, and applies every pre-registered decision rule.
Usage: python r3_analyze.py <dir with R3 model JSONs> <syndromes_26.json> [x5_claude_raw.jsonl] > r3_results.json"""
import json, re, sys
from pathlib import Path
import numpy as np

R3DIR, SYNF = Path(sys.argv[1]), sys.argv[2]; X5F = sys.argv[3] if len(sys.argv) > 3 else None
SYN = {s["name"]: s for s in json.load(open(SYNF))["syndromes"]}; NAMES = sorted(SYN)
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]
PARAMS = {"Qwen/Qwen2.5-0.5B-Instruct": 0.5, "Qwen/Qwen2.5-1.5B-Instruct": 1.5, "Qwen/Qwen2.5-3B-Instruct": 3, "Qwen/Qwen2.5-7B-Instruct": 7,
          "tiiuae/Falcon3-1B-Instruct": 1, "tiiuae/Falcon3-3B-Instruct": 3, "tiiuae/Falcon3-7B-Instruct": 7, "tiiuae/Falcon3-10B-Instruct": 10,
          "ibm-granite/granite-3.1-2b-instruct": 2, "ibm-granite/granite-3.1-8b-instruct": 8,
          "mistralai/Mistral-7B-Instruct-v0.1": 7, "BioMistral/BioMistral-7B": 7, "google/gemma-3-4b-it": 4, "google/medgemma-4b-it": 4,
          "meta-llama/Meta-Llama-3-8B-Instruct": 8, "m42-health/Llama3-Med42-8B": 8}
FAMILIES = {"Qwen2.5": [m for m in PARAMS if m.startswith("Qwen/")], "Falcon3": [m for m in PARAMS if "Falcon3" in m],
            "Granite": [m for m in PARAMS if "granite" in m]}
PAIRS = {"Mistral->BioMistral": ("mistralai/Mistral-7B-Instruct-v0.1", "BioMistral/BioMistral-7B"),
         "Gemma3->MedGemma": ("google/gemma-3-4b-it", "google/medgemma-4b-it"),
         "Llama3->Med42 (R1)": ("meta-llama/Meta-Llama-3-8B-Instruct", "m42-health/Llama3-Med42-8B")}
LOW, TEMP = "low_income__constrained", "high_income__well_resourced"
CELLS = ["low_income__constrained", "high_income__well_resourced", "low_income__well_resourced", "high_income__constrained"]
NB = 2000; RNG = np.random.default_rng(0); BOOT = [RNG.integers(0, len(NAMES), len(NAMES)) for _ in range(NB)]


def norm(s):
    s = str(s).lower()
    for a, b in SPELL:
        s = s.replace(a, b)
    for c in ("carcinoma", "malignancy", "neoplasm", "cancer"):
        s = s.replace(c, "cancer")
    return s


def rank(rl, dz):
    keys = [norm(k) for k in dz.get("keywords", [dz["disease"]])]; low = [norm(x) for x in rl]
    return next((i for i, it in enumerate(low) if any(k in it for k in keys)), None)


def per_syn(low_lists, temp_lists, k=None, keep=None):
    """-> {syndrome: {A, B, H, S}} failure indicators (paper definitions). k: truncate both lists to top-k (length-
    invariant 'top-k drop'). keep: optional set of syndromes to use (others empty). S = normalised rank shift
    (low-income rank/len minus temperate rank/len; absent = 1.0) for underreported targets present in temperate."""
    out = {}
    for n in NAMES:
        A, B, H, S = [], [], [], []
        if keep is not None and n not in keep:
            out[n] = {"A": A, "B": B, "H": H, "S": S}; continue
        ll, tl = low_lists.get(n, []), temp_lists.get(n, [])
        if k:
            ll, tl = ll[:k], tl[:k]
        for dz in SYN[n]["differential"]:
            re_, rt_ = rank(ll, dz), rank(tl, dz)
            if dz.get("underreported_in_low_income") and rt_ is not None:
                S.append((1.0 if re_ is None else re_ / max(len(ll), 1)) - rt_ / max(len(tl), 1))
            if dz["endemic_dependent"] and re_ is not None:
                A.append(1 - int((rt_ is None) or (rt_ > re_)))
            if dz.get("underreported_in_low_income") and rt_ is not None:
                B.append(1 - int((re_ is not None) and (re_ <= rt_))); H.append(int(re_ is None))
        out[n] = {"A": A, "B": B, "H": H, "S": S}
    return out


def pooled(ps, key, idx=None):
    ns = [NAMES[i] for i in idx] if idx is not None else NAMES
    v = sum((ps[n][key] for n in ns), []); return float(np.mean(v)) if v else np.nan


def ci_rate(ps, key):
    pt = pooled(ps, key); bs = np.array([pooled(ps, key, b) for b in BOOT]); bs = bs[~np.isnan(bs)]
    n = sum(len(ps[x][key]) for x in NAMES)
    if n == 0 or len(bs) == 0:
        return {"rate": None, "ci": [None, None], "n": n}
    return {"rate": round(pt, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": n}


def ci_diff(psa, psb, key):
    pt = pooled(psb, key) - pooled(psa, key); bs = np.array([pooled(psb, key, b) - pooled(psa, key, b) for b in BOOT]); bs = bs[~np.isnan(bs)]
    if len(bs) == 0 or np.isnan(pt):
        return {"diff": None, "ci": [None, None]}
    return {"diff": round(pt, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}


def incl(lists, n):
    U = [d for d in SYN[n]["differential"] if d.get("underreported_in_low_income")]
    return np.mean([rank(lists.get(n, []), d) is not None for d in U]) if U else np.nan


models = {}
for f in sorted(R3DIR.glob("*.json")):
    d = json.load(open(f))
    if "raw" in d:
        models[d["model"]] = d
res = {"models_run": {m: d["repo_used"] for m, d in models.items()}, "malformed": {m: d["malformed"] for m, d in models.items()},
       "missing": [m for m in PARAMS if m not in models]}
PS = {t: {m: per_syn(d["raw"][t][LOW], d["raw"][t][TEMP]) for m, d in models.items()} for t in ("T1", "T2", "T3")}
LEN = {m: {t: round(float(np.mean([len(v) for c in (LOW, TEMP) for v in d["raw"][t][c].values()])), 2) for t in ("T1", "T2", "T3")} for m, d in models.items()}
PS5 = {t: {m: per_syn(d["raw"][t][LOW], d["raw"][t][TEMP], k=5) for m, d in models.items()} for t in ("T1", "T2", "T3")}
def empty_frac(d, t):
    v = [len(x) == 0 for c in (LOW, TEMP) for x in d["raw"][t][c].values()]; return float(np.mean(v))
def json_only(d, t):
    ok = d.get("json_ok", {}).get(t)
    if not ok:
        return None
    keep = {n for n in NAMES if ok.get(LOW, {}).get(n) and ok.get(TEMP, {}).get(n)}
    ps = per_syn(d["raw"][t][LOW], d["raw"][t][TEMP], keep=keep)
    return {"n_syndromes": len(keep), "B": ci_rate(ps, "B"), "H": ci_rate(ps, "H")}
res["table"] = {t: {m: {"A": ci_rate(PS[t][m], "A"), "B": ci_rate(PS[t][m], "B"), "H": ci_rate(PS[t][m], "H"),
                        "H_top5": ci_rate(PS5[t][m], "H"), "rank_shift": ci_rate(PS[t][m], "S"), "len": LEN[m][t],
                        "empty_frac": round(empty_frac(d, t), 3), "unusable": empty_frac(d, t) > 0.5,
                        "json_only": json_only(d, t)}
                    for m, d in models.items()} for t in ("T1", "T2", "T3")}
UNUSABLE = {(m, t) for t in ("T1", "T2", "T3") for m in models if res["table"][t][m]["unusable"]}
res["unusable"] = sorted(f"{m} {t}" for m, t in UNUSABLE)

# R3-H1: "worst" only if its paired B diff vs every other model has CI > 0
t1B = {m: pooled(PS["T1"][m], "B") for m in models}
if t1B:
    top = max(t1B, key=t1B.get); worst_ok = all(ci_diff(PS["T1"][o], PS["T1"][top], "B")["ci"][0] > 0 for o in models if o != top)
    res["R3_H1"] = {"highest_B_T1": top, "B": round(t1B[top], 3), "worst_claim_allowed": bool(worst_ok)}

# R3-H2: within-family slope of B on log10(params), syndrome-cluster bootstrap
def slope(ms, t, idx=None):
    xs = np.log10([PARAMS[m] for m in ms]); ys = np.array([pooled(PS[t][m], "B", idx) for m in ms])
    ok = ~np.isnan(ys); return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else np.nan
h2 = {}
for fam, ms in FAMILIES.items():
    ms = [m for m in ms if m in models]
    if len(ms) < 2:
        h2[fam] = {"error": "fewer than 2 sizes ran"}; continue
    e = {}
    for t in ("T1", "T2", "T3"):
        bs = np.array([slope(ms, t, b) for b in BOOT]); bs = bs[~np.isnan(bs)]
        e[t] = {"slope": round(slope(ms, t), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    e["supported"] = bool(e["T1"]["ci"][0] > 0 and e["T2"]["slope"] > 0 and e["T3"]["slope"] > 0); h2[fam] = e
res["R3_H2"] = {"families": h2, "supported": bool(sum(v.get("supported", False) for v in h2.values()) >= 2)}

# R3-H3: medical - base, paired
h3 = {}
for k, (b, m) in PAIRS.items():
    if b in models and m in models:
        h3[k] = {t: {"B": ci_diff(PS[t][b], PS[t][m], "B"), "H": ci_diff(PS[t][b], PS[t][m], "H")} for t in ("T1", "T2", "T3")}
        h3[k]["len_base_vs_med_T1"] = [LEN[b]["T1"], LEN[m]["T1"]]
    else:
        h3[k] = {"error": "pair incomplete", "have": [x for x in (b, m) if x in models]}
lt = lambda x, y: x is not None and x < y                                        # noqa: E731  (None = unusable -> condition not met)
gt = lambda x, y: x is not None and x > y                                        # noqa: E731
red = [k for k, v in h3.items() if "T1" in v and lt(v["T1"]["B"]["ci"][1], 0) and all(lt(v[t]["B"]["diff"], 0) for t in ("T2", "T3"))]
wor = [k for k, v in h3.items() if "T1" in v and gt(v["T1"]["B"]["ci"][0], 0)]
r1 = h3.get("Llama3->Med42 (R1)", {}); r1v = None
if "T1" in r1:
    lo, hi = r1["T1"]["B"]["ci"]
    r1v = None if lo is None else ("inherited" if lo <= 0 <= hi else ("induced" if lo > 0 else "medical tuning reduced it"))
res["R3_H3"] = {"pairs": h3, "reliable_reduction_pairs": red, "worsening_pairs": wor,
                "does_not_reliably_reduce": not red, "can_worsen": bool(wor), "R1_verdict": r1v}

# R3-H4: template robustness for the >=7B open models
big = [m for m in models if PARAMS.get(m, 0) >= 7]
use = lambda t: [m for m in big if (m, t) not in UNUSABLE]                       # noqa: E731  unusable entries excluded, never imputed
res["R3_H4"] = {t: {"mean_B": round(float(np.nanmean([pooled(PS[t][m], "B") for m in use(t)])), 3),
                    "mean_top5": round(float(np.nanmean([pooled(PS5[t][m], "H") for m in use(t)])), 3),
                    "mean_H": round(float(np.nanmean([pooled(PS[t][m], "H") for m in use(t)])), 3), "models": use(t)} for t in ("T1", "T2", "T3")}
res["R3_H4"]["holds_all"] = bool(big and all(res["R3_H4"][t]["mean_B"] >= 0.5 and res["R3_H4"][t]["mean_H"] >= 0.25 for t in ("T1", "T2", "T3")))

# R3-H5: sampling variance
h5 = {}
for m, d in models.items():
    if d.get("sampled"):
        runs = [per_syn(d["sampled"][s][LOW], d["sampled"][s][TEMP]) for s in d["sampled"]]
        Bs = [pooled(r, "B") for r in runs]; Hs = [pooled(r, "H") for r in runs]
        g = (pooled(PS["T1"][m], "B"), pooled(PS["T1"][m], "H"))
        h5[m] = {"B_runs": [round(x, 3) for x in Bs], "H_runs": [round(x, 3) for x in Hs], "B_sd": round(float(np.std(Bs, ddof=1)), 3),
                 "H_sd": round(float(np.std(Hs, ddof=1)), 3), "greedy_B_H": [round(g[0], 3), round(g[1], 3)],
                 "greedy_representative": bool(min(Bs) - 0.05 <= g[0] <= max(Bs) + 0.05 and min(Hs) - 0.05 <= g[1] <= max(Hs) + 0.05)}
res["R3_H5"] = h5

# R3-X1: factorial on validated syndromes, models >= 3B
def x1(d, t):
    M = np.array([[incl(d["raw"][t][c], n) for c in CELLS] for n in NAMES], float)   # cols: LIC-C, HIC-WR, LIC-WR, HIC-C
    keep = ~np.isnan(M).any(1); M = M[keep]
    g = (M[:, 1] + M[:, 3] - M[:, 0] - M[:, 2]) / 2; r = (M[:, 1] + M[:, 2] - M[:, 0] - M[:, 3]) / 2
    it = (M[:, 1] - M[:, 3]) - (M[:, 2] - M[:, 0]); rng = np.random.default_rng(1); out = {"n": int(len(M)),
        "cells": dict(zip(CELLS, np.round(M.mean(0), 3).tolist()))}
    for k, v in (("geography_HIC_minus_LIC", g), ("resource_WR_minus_C", r), ("interaction", it)):
        bs = [rng.choice(v, len(v)).mean() for _ in range(NB)]
        out[k] = [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    return out
X1 = {m: {t: x1(d, t) for t in ("T1", "T2", "T3") if (m, t) not in UNUSABLE} for m, d in models.items()
      if PARAMS.get(m, 0) >= 3 and (m, "T1") not in UNUSABLE}
nres = sum(v["T1"]["resource_WR_minus_C"][1] > 0 and v["T1"]["geography_HIC_minus_LIC"][1] <= 0 <= v["T1"]["geography_HIC_minus_LIC"][2] for v in X1.values())
ngeo = sum(v["T1"]["geography_HIC_minus_LIC"][1] > 0 and v["T1"]["resource_WR_minus_C"][1] <= 0 <= v["T1"]["resource_WR_minus_C"][2] for v in X1.values())
k = len(X1)
res["R3_X1"] = {"per_model": X1, "n_models": k, "n_resource_pattern": int(nres), "n_geography_pattern": int(ngeo),
                "verdict": ("resource" if k and nres >= 2 * k / 3 else "geography" if k and ngeo >= 2 * k / 3 else "both/mixed")}

# R3-X5: Claude repeats
if X5F and Path(X5F).exists():
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
    x5 = {}
    for rep, d in sorted(runs.items()):
        ps = per_syn(d["low"], d["temp"]); ps5 = per_syn(d["low"], d["temp"], k=5)
        x5[rep] = {"A": ci_rate(ps, "A"), "B": ci_rate(ps, "B"), "H": ci_rate(ps, "H"), "H_top5": ci_rate(ps5, "H"),
                   "rank_shift": ci_rate(ps, "S"), "mean_len": round(float(np.mean([len(v) for v in list(d["low"].values()) + list(d["temp"].values())])), 2),
                   "n_low": len(d["low"]), "n_temp": len(d["temp"])}
    res["R3_X5"] = {"model_ids": sorted({m for r in recs for m in r["models"]}), "errors": sum(bool(r["is_error"]) for r in recs),
                    "runs": x5, "band_holds": bool(len(x5) == 3 and all(v["B"]["rate"] >= 0.70 for v in x5.values()))}
print(json.dumps(res, indent=1))
