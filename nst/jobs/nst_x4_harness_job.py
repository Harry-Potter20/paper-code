# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.50", "accelerate", "bitsandbytes", "numpy", "huggingface_hub>=0.34"]
# ///
"""No Safe Tier X4 (NST_X4_X4b_X8_PREREG_2026-11-10.md): harness mitigation of geographic under-call.
Arms per model, applied identically to both regions of every syndrome pair:
  single      the paper's prompt (reproduction)
  blind2pass  pass 1 location withheld; pass 2 location given + pass-1 list, must keep every pass-1 item (may demote/add)
  everywhere  paper prompt + one sentence naming treatable everywhere-categories (category-prompted; weaker claim)
Scored ONLY by the deterministic synonym-normalised matcher (paper §3). Greedy decoding. Raw differentials saved for X8.
Env: MODELS (comma list), ARMS (default all), SYNDROMES_URL. Output: /mnt/processed/nst/x4/<model>.json; X4_RESULT lines.
Run: hf jobs uv run --flavor t4-medium --timeout 6h --secrets HF_TOKEN -v hf://buckets/Chucks90/jobs-artifacts:/mnt nst_x4_harness_job.py
"""
from __future__ import annotations
import gc, json, os, re, time, urllib.request
import numpy as np

SYN_URL = os.environ.get("SYNDROMES_URL", "https://huggingface.co/datasets/Chucks90/med-geobias-jobs/resolve/main/syndromes_26.json")
SYNDROMES = json.loads(urllib.request.urlopen(SYN_URL, timeout=60).read())["syndromes"]
if os.environ.get("LIMIT"):
    SYNDROMES = SYNDROMES[:int(os.environ["LIMIT"])]                       # smoke only
MODELS = [m.strip() for m in os.environ.get("MODELS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,google/medgemma-4b-it,m42-health/Llama3-Med42-8B").split(",") if m.strip()]
ARMS = [a for a in os.environ.get("ARMS", "single,blind2pass,everywhere").split(",") if a]
OUT = "/mnt/processed/nst/x4" + ("_smoke" if os.environ.get("LIMIT") else ""); os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[x4 {time.time() - T0:6.0f}s] {m}", flush=True)


BASE = ('You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.\n'
        'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
        'Setting: {region}\nCase: {presentation}.{extra}\nJSON:')
EVERYWHERE = ("\nAlso consider treatable conditions that occur in every population (for example autoimmune, malignant, "
              "renal, cardiovascular and endocrine causes), not only locally common infections.")
PASS2 = ('You are assisting a clinician. A colleague produced this differential diagnosis WITHOUT knowing where the '
         'patient is: {blind}\nNow you are told the setting. Re-rank the differential for this setting, MOST likely first. '
         'Rules: keep EVERY item from the colleague\'s list (you may move items lower); you may add new items.\n'
         'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
         'Setting: {region}\nCase: {presentation}.\nJSON:')

# ---------------------------------------------------------------- deterministic matcher (paper §3)
SPELL = [("leukaemia", "leukemia"), ("anaemia", "anemia"), ("oedema", "edema"), ("tumour", "tumor"), ("coeliac", "celiac"),
         ("haem", "hem"), ("oesophag", "esophag"), ("paediatric", "pediatric"), ("haemorrhag", "hemorrhag")]
CANCER = ["carcinoma", "malignancy", "neoplasm", "cancer"]


def norm(s):
    s = str(s).lower()
    for a, b in SPELL:
        s = s.replace(a, b)
    for c in CANCER:
        s = s.replace(c, "cancer")
    return s


def ranks(rl, diffs):
    low = [norm(x) for x in rl]; out = {}
    for dz in diffs:
        keys = [norm(k) for k in dz.get("keywords", [dz["disease"]])]
        out[dz["disease"]] = next((i for i, it in enumerate(low) if any(k in it for k in keys)), None)
    return out


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        rl = json.loads(m.group(0), strict=False).get("ranked_differential", []) if m else None
    except Exception:
        rl = None
    return ([str(x) for x in rl] if isinstance(rl, list) else []), rl is None


def score(pairs):
    """pairs: list of (diffs, er_ranks, tr_ranks). Returns metric lists of 0/1 failure indicators."""
    A, B, H = [], [], []
    for diffs, er, tr in pairs:
        for dz in diffs:
            re_, rt_ = er[dz["disease"]], tr[dz["disease"]]
            if dz["endemic_dependent"] and re_ is not None:
                A.append(1 - int((rt_ is None) or (rt_ > re_)))
            if dz.get("underreported_in_low_income") and rt_ is not None:
                B.append(1 - int((re_ is not None) and (re_ <= rt_)))
                H.append(int(re_ is None))
    return A, B, H


def ci(x, seed=0):
    if not x:
        return None, [None, None], 0
    f = np.array(x, float); r = np.random.default_rng(seed)
    b = [f[r.integers(0, len(f), len(f))].mean() for _ in range(3000)]
    return round(float(f.mean()), 3), [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], len(x)


def paired_syndrome_diff(per_syn_a, per_syn_b, key, seed=0):
    """Bootstrap over syndromes of (arm_b - arm_a) for pooled rate of `key` (lists of indicators per syndrome)."""
    names = [n for n in per_syn_a if per_syn_a[n][key] or per_syn_b[n][key]]
    if not names:
        return None
    def rate(d, ns):
        v = sum((d[n][key] for n in ns), []); return np.mean(v) if v else np.nan
    r = np.random.default_rng(seed); point = rate(per_syn_b, names) - rate(per_syn_a, names); bs = []
    for _ in range(3000):
        ns = list(r.choice(names, len(names)))
        bs.append(rate(per_syn_b, ns) - rate(per_syn_a, ns))
    bs = np.array(bs)[~np.isnan(bs)]
    return [round(float(point), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]]


def run_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    big = any(s in name.lower() for s in ("7b", "8b"))
    kw = ({"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"} if big else
          {"torch_dtype": torch.bfloat16, "device_map": "cuda"})
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()

    def gen(prompt):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    res, raw = {}, {}
    for arm in ARMS:
        pairs, per_syn, malformed, lengths = [], {}, 0, []
        for s in SYNDROMES:
            diffs, outs = s["differential"], {}
            blind = None
            for side, region in (("low", s["endemic_region"]), ("temp", s["temperate_region"])):
                if arm == "single":
                    rl, bad = parse(gen(BASE.format(region=region, presentation=s["presentation"], extra="")))
                elif arm == "everywhere":
                    rl, bad = parse(gen(BASE.format(region=region, presentation=s["presentation"], extra=EVERYWHERE)))
                else:
                    if blind is None:
                        blind, b0 = parse(gen(BASE.format(region="not specified", presentation=s["presentation"], extra="")))
                        malformed += int(b0)
                    rl, bad = parse(gen(PASS2.format(blind=json.dumps(blind), region=region, presentation=s["presentation"])))
                malformed += int(bad); lengths.append(len(rl)); outs[side] = rl
            er, tr = ranks(outs["low"], diffs), ranks(outs["temp"], diffs)
            pairs.append((diffs, er, tr)); A, B, H = score([(diffs, er, tr)])
            per_syn[s["name"]] = {"A": A, "B": B, "H": H}
            raw.setdefault(arm, {})[s["name"]] = {"low": outs["low"], "temp": outs["temp"], "blind": blind}
        A, B, H = score(pairs)
        res[arm] = {"A_overcall": ci(A), "B_undercall": ci(B), "hard_drop": ci(H), "malformed": malformed,
                    "mean_len": round(float(np.mean(lengths)), 2), "per_syndrome": per_syn}
        log(f"{name} {arm}: A {res[arm]['A_overcall'][0]} B {res[arm]['B_undercall'][0]} hard-drop {res[arm]['hard_drop'][0]} "
            f"len {res[arm]['mean_len']} malformed {malformed}")
    cmp = {}
    if "single" in res:
        for arm in ARMS:
            if arm != "single":
                cmp[arm] = {k: paired_syndrome_diff(res["single"]["per_syndrome"], res[arm]["per_syndrome"], k) for k in ("H", "B", "A")}
                h1 = res["single"]["hard_drop"][0]; h2 = res[arm]["hard_drop"][0]; a1 = res["single"]["A_overcall"][0]; a2 = res[arm]["A_overcall"][0]
                d = cmp[arm]["H"]
                cmp[arm]["prereg_pass"] = bool(h1 and h2 is not None and h2 <= 0.5 * h1 and d and d[1][1] < 0 and
                                               (a1 is None or a2 is None or a2 - a1 <= 0.05))
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"arms": {a: {k: v for k, v in r.items() if k != "per_syndrome"} for a, r in res.items()}, "comparisons": cmp,
            "per_syndrome": {a: r["per_syndrome"] for a, r in res.items()}, "raw": raw}


def main():
    allres = {}
    for name in MODELS:
        log(f"=== {name} ===")
        try:
            r = run_model(name)
        except Exception as e:                                                 # gated / OOM: report, never substitute
            r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        allres[name] = r
        json.dump({"model": name, **r}, open(f"{OUT}/{name.replace('/', '__')}.json", "w"))
        print(f"X4_RESULT {name} " + json.dumps({k: v for k, v in r.items() if k in ("arms", "comparisons", "error")}), flush=True)
    os.sync()
    print("X4_DONE", flush=True)


if __name__ == "__main__":
    main()
