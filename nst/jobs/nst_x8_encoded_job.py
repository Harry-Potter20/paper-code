# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.50", "accelerate", "bitsandbytes", "numpy", "scipy", "scikit-learn", "huggingface_hub>=0.34"]
# ///
"""No Safe Tier X8 (NST_X4_X4b_X8_PREREG_2026-11-10.md + Amendment A1): geographically encoded but not emitted?
Uses X4's single-arm differentials (/mnt/processed/nst/x4/<model>.json). For each underreported target present in the
temperate differential:
  readout 1: logistic probe on [h_L(final prompt token) ; mean input embedding of target name], trained on TEMPERATE
             prompts of TRAIN syndromes (present targets vs absent distractors), applied to LOW-INCOME prompts of TEST
             syndromes where the target was hard-dropped. 'Encoded' = probe score > 95th pct of temperate negatives.
  readout 2: log P(target as next JSON item | low-income prompt + generated list) vs a length-matched absent distractor.
Kill checks: partial Spearman of probe score with hard-drop controlling for ||h_L|| and base log P(target | 'A diagnosis:');
random-init null (RANDOM_NULL=1, smallest model only). Output /mnt/processed/nst/x8/<model>.json; X8_RESULT lines.
Run: hf jobs uv run --flavor t4-medium --timeout 4h --secrets HF_TOKEN -v hf://buckets/Chucks90/jobs-artifacts:/mnt nst_x8_encoded_job.py
"""
from __future__ import annotations
import json, os, re, time, urllib.request
import numpy as np

SYN_URL = "https://huggingface.co/datasets/Chucks90/med-geobias-jobs/resolve/main/syndromes_26.json"
SYNDROMES = json.loads(urllib.request.urlopen(SYN_URL, timeout=60).read())["syndromes"]
NAMES = sorted(s["name"] for s in SYNDROMES)
TEST = {n for i, n in enumerate(NAMES) if i % 5 in (1, 3)}                     # frozen split (prereg)
MODELS = [m for m in os.environ.get("MODELS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,google/medgemma-4b-it,m42-health/Llama3-Med42-8B").split(",") if m]
RANDOM_NULL = os.environ.get("RANDOM_NULL") == "1"
X4 = "/mnt/processed/nst/x4"; OUT = "/mnt/processed/nst/x8"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
BASE = ('You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.\n'
        'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
        'Setting: {region}\nCase: {presentation}.\nJSON:')
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]


def log(m):
    print(f"[x8 {time.time() - T0:6.0f}s] {m}", flush=True)


def norm(s):
    s = str(s).lower()
    for a, b in SPELL:
        s = s.replace(a, b)
    for c in ("carcinoma", "malignancy", "neoplasm", "cancer"):
        s = s.replace(c, "cancer")
    return s


def present(rl, dz):
    keys = [norm(k) for k in dz.get("keywords", [dz["disease"]])]
    return any(any(k in norm(x) for k in keys) for x in rl)


def run(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr, wilcoxon
    raw = json.load(open(f"{X4}/{name.replace('/', '__')}.json"))["raw"]["single"]
    tok = AutoTokenizer.from_pretrained(name)
    big = any(s in name.lower() for s in ("7b", "8b"))
    kw = ({"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"} if big else
          {"torch_dtype": torch.bfloat16, "device_map": "cuda"})
    if RANDOM_NULL:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), torch_dtype=torch.bfloat16).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    emb = model.get_input_embeddings()

    def chat(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def hidden(prompt):
        enc = tok(chat(prompt), return_tensors="pt").to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs])          # (L+1, d)

    @torch.no_grad()
    def name_emb(t):
        ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        return emb(ids)[0].float().mean(0).cpu().numpy()

    @torch.no_grad()
    def logp_cont(prefix, cont):
        a = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        b = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([a, b], 1).to(model.device)
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]; n = b.shape[1]
        return float(lp[-n:].gather(1, tgt[-n:, None]).sum())

    alld = sorted({dz["disease"] for s in SYNDROMES for dz in s["differential"]})
    rows = []
    for s in SYNDROMES:
        r = raw.get(s["name"]);
        if not r:
            continue
        h_low, h_temp = hidden(BASE.format(region=s["endemic_region"], presentation=s["presentation"])), \
                        hidden(BASE.format(region=s["temperate_region"], presentation=s["presentation"]))
        here = {dz["disease"] for dz in s["differential"]}
        pool = [d for d in alld if d not in here]
        for dz in s["differential"]:
            if not dz.get("underreported_in_low_income"):
                continue
            t = dz["disease"]; in_t, in_l = present(r["temp"], dz), present(r["low"], dz)
            dist = min(pool, key=lambda d: (abs(len(tok(d).input_ids) - len(tok(t).input_ids)), d))
            rows.append({"syn": s["name"], "target": t, "distractor": dist, "in_temp": in_t, "in_low": in_l,
                         "hard_drop": bool(in_t and not in_l), "test": s["name"] in TEST,
                         "h_low": h_low, "h_temp": h_temp, "e_t": name_emb(t), "e_d": name_emb(dist),
                         "low_prefix": chat(BASE.format(region=s["endemic_region"], presentation=s["presentation"])) +
                                       '{"ranked_differential": [' + ", ".join(json.dumps(x) for x in r["low"]) + (", " if r["low"] else ""),
                         "base_t": logp_cont("A diagnosis:", " " + t)})
    L = rows[0]["h_low"].shape[0]
    # readout 1: choose layer on TRAIN by CV AUROC (temperate present vs distractor)
    from sklearn.model_selection import cross_val_score
    def Xy(rs, which):
        X, y = [], []
        for q in rs:
            if q["in_temp"]:
                X.append(np.r_[q[which][lay], q["e_t"]]); y.append(1)
                X.append(np.r_[q[which][lay], q["e_d"]]); y.append(0)
        return np.array(X), np.array(y)
    train = [q for q in rows if not q["test"]]; test = [q for q in rows if q["test"]]
    best = (-1, None)
    for lay in range(2, L, max(1, L // 12)):
        X, y = Xy(train, "h_temp")
        if len(set(y)) < 2:
            continue
        auc = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=4, scoring="roc_auc").mean()
        best = max(best, (auc, lay))
    auc_tr, lay = best; log(f"{name}: layer {lay} train CV AUROC {auc_tr:.3f}")
    X, y = Xy(train, "h_temp"); clf = LogisticRegression(max_iter=4000, C=0.1).fit(X, y)
    neg = clf.decision_function(np.array([np.r_[q["h_temp"][lay], q["e_d"]] for q in train if q["in_temp"]]))
    thr = float(np.percentile(neg, 95))
    drops = [q for q in test if q["hard_drop"]]
    enc_scores = clf.decision_function(np.array([np.r_[q["h_low"][lay], q["e_t"]] for q in drops])) if drops else np.array([])
    frac_encoded = float((enc_scores > thr).mean()) if drops else None
    # readout 2: log-prob target vs distractor as next list item
    lp_t = [logp_cont(q["low_prefix"], json.dumps(q["target"])) for q in drops]
    lp_d = [logp_cont(q["low_prefix"], json.dumps(q["distractor"])) for q in drops]
    wil = wilcoxon(lp_t, lp_d, alternative="greater") if len(drops) >= 6 else None
    # kill check: partial Spearman (probe score vs hard_drop | ||h||, base logp) on TEST targets present in temperate
    cand = [q for q in test if q["in_temp"]]
    sc = clf.decision_function(np.array([np.r_[q["h_low"][lay], q["e_t"]] for q in cand])) if cand else np.array([])
    def resid(v, Z):
        Z = np.c_[np.ones(len(Z)), Z]; return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    if len(cand) >= 8:
        Z = np.c_[[np.linalg.norm(q["h_low"][lay]) for q in cand], [q["base_t"] for q in cand]]
        hd = np.array([q["hard_drop"] for q in cand], float)
        ps = spearmanr(resid(sc, Z), resid(hd, Z))
        partial = [round(float(ps.correlation), 3), round(float(ps.pvalue), 4)]
    else:
        partial = None
    out = {"model": name, "random_null": RANDOM_NULL, "layer": int(lay), "train_cv_auroc": round(float(auc_tr), 3),
           "n_rows": len(rows), "n_test_hard_drops": len(drops), "frac_encoded": None if frac_encoded is None else round(frac_encoded, 3),
           "readout2_median_lp_target_minus_distractor": round(float(np.median(np.array(lp_t) - np.array(lp_d))), 3) if drops else None,
           "readout2_wilcoxon_p": None if wil is None else round(float(wil.pvalue), 5),
           "partial_spearman_score_vs_harddrop": partial,
           "claim_encoded_not_emitted": bool(frac_encoded is not None and frac_encoded >= 0.5 and wil is not None and wil.pvalue < 0.01)}
    del model; torch.cuda.empty_cache()
    return out


def main():
    for m in MODELS:
        try:
            r = run(m)
        except Exception as e:
            r = {"model": m, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        json.dump(r, open(f"{OUT}/{m.replace('/', '__')}{'_randomnull' if RANDOM_NULL else ''}.json", "w"))
        print("X8_RESULT " + json.dumps(r), flush=True)
    print("X8_DONE", flush=True)


if __name__ == "__main__":
    main()
