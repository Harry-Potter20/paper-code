import json, sys, re, numpy as np
B = "/Users/user/Elysium/Med_Architecture_Research/clinical_safety_eval"; X = sys.argv[1]
sys.path.insert(0, f"{B}/jobs")
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]
def norm(s):
    s = str(s).lower()
    for a, b in SPELL: s = s.replace(a, b)
    for c in ("carcinoma", "malignancy", "neoplasm", "cancer"): s = s.replace(c, "cancer")
    return s
def has(rl, dz):
    keys = [norm(k) for k in dz.get("keywords", [dz["disease"]])]
    return any(any(k in norm(x) for k in keys) for x in rl)
syn26 = json.load(open(f"{B}/x6/../x6/syndromes_26.json"))["syndromes"] if False else None
import glob
s26 = json.load(open(glob.glob(f"{B}/**/syndromes_26.json", recursive=True)[0]))["syndromes"]
s24 = json.load(open(f"{B}/x6/syndromes_x6_draft.json"))["syndromes"]
orig = {s["name"] for s in s26}; SY = s26 + s24
CELLS = ["low_income__well_resourced", "low_income__constrained", "high_income__well_resourced", "high_income__constrained"]
out = {}
for mf in ("Qwen__Qwen2.5-7B-Instruct", "m42-health__Llama3-Med42-8B"):
    raw = json.load(open(f"{X}/{mf}_raw200.json"))
    rows = []   # per syndrome: (orig?, per-cell under-incl, per-cell endemic-incl, mean len per cell)
    for s in SY:
        r = raw.get(s["name"])
        if not r: continue
        U = [d for d in s["differential"] if d.get("underreported_in_low_income")]
        E = [d for d in s["differential"] if d.get("endemic_dependent")]
        u = [np.mean([has(r[c], d) for d in U]) if U else np.nan for c in CELLS]
        e = [np.mean([has(r[c], d) for d in E]) if E else np.nan for c in CELLS]
        rows.append((s["name"] in orig, u, e, [len(r[c]) for c in CELLS]))
    def dec(M):   # M: n x 4 (LIC-WR, LIC-C, HIC-WR, HIC-C); returns geography (HIC-LIC), resource (WR-C), interaction
        M = M[~np.isnan(M).any(1)]
        g = (M[:, 2] + M[:, 3] - M[:, 0] - M[:, 1]) / 2; rs = (M[:, 0] + M[:, 2] - M[:, 1] - M[:, 3]) / 2
        it = (M[:, 2] - M[:, 3]) - (M[:, 0] - M[:, 1])
        rng = np.random.default_rng(0); res = {"n": len(M), "cells": dict(zip(CELLS, np.round(M.mean(0), 3).tolist()))}
        for k, v in (("geography_HIC_minus_LIC", g), ("resource_WR_minus_C", rs), ("interaction", it)):
            bs = [rng.choice(v, len(v)).mean() for _ in range(4000)]
            res[k] = [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        return res
    o = {}
    for sub, f in (("all50", lambda x: True), ("orig26", lambda x: x[0]), ("x6_24", lambda x: not x[0])):
        R = [x for x in rows if f(x)]
        o[sub] = {"underreported_incl": dec(np.array([x[1] for x in R], float)), "endemic_incl": dec(np.array([x[2] for x in R], float)),
                  "mean_len": dict(zip(CELLS, np.round(np.mean([x[3] for x in R], 0), 2).tolist()))}
    out[mf] = o
json.dump(out, open(f"{X}/x1_provisional.json", "w"), indent=1)
for m, o in out.items():
    print("==", m)
    for sub, v in o.items():
        print(" ", sub, "| UNDER", v["underreported_incl"], "\n     ENDEMIC", v["endemic_incl"], "\n     LEN", v["mean_len"])
