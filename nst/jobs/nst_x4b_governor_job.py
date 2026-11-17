# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.50", "accelerate", "bitsandbytes", "numpy", "scikit-learn", "huggingface_hub>=0.34"]
# ///
"""No Safe Tier X4b (NST_X4_X4b_X8_PREREG_2026-11-10.md; programme Idea A): dual-controlled activation governor.
Stage 1 (sensor, RQ-A1): probe on [h_L(final prompt token of the LOW-INCOME prompt) ; mean target-name embedding] predicting
  whether each underreported target (present in the temperate differential, per X4 single-arm outputs) is HARD-DROPPED.
  Layer chosen by TRAIN-split CV AUROC; evaluated on TEST syndromes. KILL if TEST AUROC < 0.70 (no steering is run).
Stage 2 (only if the sensor passes): direction d = mean(h | location-blind prompt) - mean(h | low-income prompt) at layer L
  (TRAIN syndromes). Arms on TEST syndromes, low-income prompts (temperate side reused from X4 single):
    fixed alpha in {0.5,1,2,4,8} x ||d||-normalised;  random direction (alpha=4);  governor eps in {0.05,0.1,0.2,0.3}:
    alpha_{t+1} = [alpha_t + eta (p_t - eps)]+, p_t = max over the syndrome's targets of the probe on the current token state.
  Metrics: hard-drop, under-call, over-call (endemic targets), mean length — deterministic matcher (as X4).
Output /mnt/processed/nst/x4b/<model>.json; X4B_RESULT lines.
Run: hf jobs uv run --flavor a10g-large --timeout 6h --secrets HF_TOKEN -v hf://buckets/Chucks90/jobs-artifacts:/mnt nst_x4b_governor_job.py
"""
from __future__ import annotations
import json, os, re, time, urllib.request
import numpy as np

SYN_URL = "https://huggingface.co/datasets/Chucks90/med-geobias-jobs/resolve/main/syndromes_26.json"
SYNDROMES = json.loads(urllib.request.urlopen(SYN_URL, timeout=60).read())["syndromes"]
NAMES = sorted(s["name"] for s in SYNDROMES); TEST = {n for i, n in enumerate(NAMES) if i % 5 in (1, 3)}
MODELS = [m for m in os.environ.get("MODELS", "Qwen/Qwen2.5-7B-Instruct,m42-health/Llama3-Med42-8B").split(",") if m]
X4 = "/mnt/processed/nst/x4"; OUT = "/mnt/processed/nst/x4b"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
BASE = ('You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.\n'
        'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
        'Setting: {region}\nCase: {presentation}.\nJSON:')
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]


def log(m):
    print(f"[x4b {time.time() - T0:6.0f}s] {m}", flush=True)


def norm(s):
    s = str(s).lower()
    for a, b in SPELL:
        s = s.replace(a, b)
    for c in ("carcinoma", "malignancy", "neoplasm", "cancer"):
        s = s.replace(c, "cancer")
    return s


def rank_of(rl, dz):
    keys = [norm(k) for k in dz.get("keywords", [dz["disease"]])]
    return next((i for i, it in enumerate([norm(x) for x in rl]) if any(k in it for k in keys)), None)


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        rl = json.loads(m.group(0), strict=False).get("ranked_differential", []) if m else []
    except Exception:
        rl = []
    return [str(x) for x in rl] if isinstance(rl, list) else []


def run(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    raw = json.load(open(f"{X4}/{name.replace('/', '__')}.json"))["raw"]["single"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto").eval()
    layers = model.model.layers; emb = model.get_input_embeddings()
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)  # noqa: E731

    @torch.no_grad()
    def hid(prompt):
        enc = tok(chat(prompt), return_tensors="pt").to(model.device)
        return np.stack([h[0, -1].float().cpu().numpy() for h in model(**enc, output_hidden_states=True).hidden_states])

    @torch.no_grad()
    def e_name(t):
        ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        return emb(ids)[0].float().mean(0).cpu().numpy()

    inst, H_low, H_blind = [], {}, {}
    for s in SYNDROMES:
        r = raw.get(s["name"])
        if not r:
            continue
        H_low[s["name"]] = hid(BASE.format(region=s["endemic_region"], presentation=s["presentation"]))
        H_blind[s["name"]] = hid(BASE.format(region="not specified", presentation=s["presentation"]))
        for dz in s["differential"]:
            if dz.get("underreported_in_low_income") and rank_of(r["temp"], dz) is not None:
                inst.append({"syn": s["name"], "dz": dz, "e": e_name(dz["disease"]), "drop": rank_of(r["low"], dz) is None,
                             "test": s["name"] in TEST})
    tr = [x for x in inst if not x["test"]]; te = [x for x in inst if x["test"]]
    L = next(iter(H_low.values())).shape[0]
    def XY(rows, lay):
        return np.array([np.r_[H_low[x["syn"]][lay], x["e"]] for x in rows]), np.array([x["drop"] for x in rows], int)
    best = (-1.0, None)
    for lay in range(2, L, max(1, L // 12)):
        X, y = XY(tr, lay)
        if len(set(y)) < 2 or min(np.bincount(y)) < 3:
            continue
        best = max(best, (cross_val_score(LogisticRegression(max_iter=3000, C=0.1), X, y, cv=3, scoring="roc_auc").mean(), lay))
    if best[1] is None:
        return {"model": name, "stage": "sensor", "error": "too few train positives for CV", "n_train": len(tr), "n_test": len(te)}
    lay = best[1]; X, y = XY(tr, lay); clf = LogisticRegression(max_iter=4000, C=0.1).fit(X, y)
    Xt, yt = XY(te, lay)
    auc = float(roc_auc_score(yt, clf.decision_function(Xt))) if len(set(yt)) == 2 else None
    res = {"model": name, "layer": int(lay), "train_cv_auroc": round(float(best[0]), 3), "n_train": len(tr), "n_test": len(te),
           "test_pos": int(yt.sum()) if len(yt) else 0, "sensor_test_auroc": None if auc is None else round(auc, 3)}
    if auc is None or auc < 0.70:
        res["stage2"] = "NOT RUN (pre-registered kill: sensor TEST AUROC < 0.70 or undefined)"
        return res
    # ---------------- stage 2: steering
    d = np.mean([H_blind[n][lay] - H_low[n][lay] for n in H_low if n not in TEST], 0)
    d_t = torch.tensor(d / (np.linalg.norm(d) + 1e-8), dtype=torch.float16, device=model.device)
    rnd = np.random.default_rng(0).standard_normal(d.shape); r_t = torch.tensor(rnd / np.linalg.norm(rnd), dtype=torch.float16, device=model.device)
    W = torch.tensor(clf.coef_[0][:d.shape[0]], dtype=torch.float32, device=model.device); b0 = float(clf.intercept_[0])
    state = {"alpha": 0.0, "dir": d_t, "mode": "off", "eps": 0.1, "eta": 2.0, "ebias": None, "alphas": []}
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["mode"] == "gov":
            last = h[0, -1].float()
            p = torch.sigmoid((last @ W) + state["ebias"]).max().item()
            state["alpha"] = max(0.0, state["alpha"] + state["eta"] * (p - state["eps"])); state["alphas"].append(state["alpha"])
        if state["mode"] != "off" and state["alpha"] > 0:
            h = h + state["alpha"] * state["dir"]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = layers[lay - 1].register_forward_hook(hook)     # hidden_states[lay] is the output of layer lay-1
    def gen(prompt):
        enc = tok(chat(prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        return parse(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    def score_arm():
        A, B, HD, lens = [], [], [], []
        for s in SYNDROMES:
            if s["name"] not in TEST or s["name"] not in raw:
                continue
            tgt_e = [e_name(dz["disease"]) for dz in s["differential"] if dz.get("underreported_in_low_income")]
            if state["mode"] == "gov":
                state["alpha"] = 0.0
                state["ebias"] = torch.tensor([b0 + float(clf.coef_[0][d.shape[0]:] @ e) for e in tgt_e] or [b0], device=model.device)
            low = gen(BASE.format(region=s["endemic_region"], presentation=s["presentation"])); temp = raw[s["name"]]["temp"]; lens.append(len(low))
            for dz in s["differential"]:
                rl, rt = rank_of(low, dz), rank_of(temp, dz)
                if dz["endemic_dependent"] and rl is not None:
                    A.append(int(not ((rt is None) or (rt > rl))))
                if dz.get("underreported_in_low_income") and rt is not None:
                    B.append(int(not ((rl is not None) and (rl <= rt)))); HD.append(int(rl is None))
        m = lambda v: round(float(np.mean(v)), 3) if v else None                       # noqa: E731
        return {"hard_drop": m(HD), "under_call": m(B), "over_call": m(A), "n_targets": len(HD), "mean_len": m(lens)}
    arms = {}
    state["mode"] = "off"; arms["none"] = score_arm()
    for a in (0.5, 1, 2, 4, 8):
        state.update(mode="fixed", alpha=float(a), dir=d_t); arms[f"fixed_{a}"] = score_arm()
    state.update(mode="fixed", alpha=4.0, dir=r_t); arms["random_4"] = score_arm()
    for e in (0.05, 0.1, 0.2, 0.3):
        state.update(mode="gov", eps=e, dir=d_t, alphas=[]); arms[f"gov_{e}"] = score_arm(); arms[f"gov_{e}"]["mean_alpha"] = round(float(np.mean(state["alphas"] or [0])), 3)
    handle.remove(); res["arms"] = arms
    return res


def main():
    for m in MODELS:
        try:
            r = run(m)
        except Exception as e:
            r = {"model": m, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        json.dump(r, open(f"{OUT}/{m.replace('/', '__')}.json", "w"))
        print("X4B_RESULT " + json.dumps(r), flush=True)
    print("X4B_DONE", flush=True)


if __name__ == "__main__":
    main()
