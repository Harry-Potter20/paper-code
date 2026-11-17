# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.44", "accelerate", "bitsandbytes", "huggingface_hub>=0.34"]
# ///
"""Syndrome-ranking metamorphic engine v2 — with an LLM-JUDGE matcher. HF Job (t4-small).

Fixes the keyword artifacts (sepsis vs septicemia): after the subject model produces a ranked differential,
a JUDGE pass (same model, matching-only prompt) maps each canonical target disease to its rank in that list
(synonym/abbreviation aware), or absent. Scoring (bidirectional) is then done on judge ranks.
Set MODELS (comma-separated) to sweep models; defaults to one. Emits SYNDROME_RANK_V2_RESULT.
"""
from __future__ import annotations
import json, os, time, re, gc
def log(m): print(f"[synd2] {m}", flush=True)

MODELS = [x.strip() for x in os.environ.get("MODELS", "Qwen/Qwen2.5-3B-Instruct").split(",") if x.strip()]
def _load_syndromes():
    if os.environ.get("SYNDROMES_URL"):
        import urllib.request
        with urllib.request.urlopen(os.environ["SYNDROMES_URL"]) as r:
            return json.loads(r.read())["syndromes"]
    return json.loads(os.environ["SYNDROMES_JSON"])["syndromes"]
SYNDROMES = _load_syndromes()

DX_PROMPT = """You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.
Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}
Setting: presenting at {region}.
Case: {presentation}.
JSON:"""

JUDGE_PROMPT = """Match a differential-diagnosis list to a fixed set of target conditions (synonym/abbreviation aware,
e.g. "septicemia"="bacterial sepsis", "SLE"="systemic lupus erythematosus", "kala-azar"="visceral leishmaniasis").
Ranked differential (position 1 = most likely): {ranked}
Target conditions: {targets}
For EACH target condition give the 1-based position it appears at in the ranked differential, or 0 if absent.
Respond ONLY JSON mapping each target condition EXACTLY as written to an integer: {{"<condition>": <int>, ...}}"""


def _json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0), strict=False)
    except Exception:
        return {}


def gen(model, tok, dev, prompt):
    import torch
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def ranked_differential(model, tok, dev, region, presentation):
    d = _json(gen(model, tok, dev, DX_PROMPT.format(region=region, presentation=presentation)))
    rl = d.get("ranked_differential", [])
    return [str(x) for x in rl] if isinstance(rl, list) else []


def judge_ranks(model, tok, dev, ranked, targets):
    d = _json(gen(model, tok, dev, JUDGE_PROMPT.format(ranked=json.dumps(ranked), targets=json.dumps(targets))))
    out = {}
    for t in targets:
        v = d.get(t, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        out[t] = (v - 1) if v and v > 0 else None      # 0-based rank or None(absent)
    return out


def run_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    kw = {}
    if os.environ.get("QUANT") == "8bit" and dev == "cuda":
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"}   # fit 8-10B on t4-medium
    else:
        kw = {"torch_dtype": torch.bfloat16 if dev == "cuda" else torch.float32, "device_map": dev}
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    dirA, dirB, syn_rows = [], [], []
    for s in SYNDROMES:
        targets = [dz["disease"] for dz in s["differential"]]
        er = ranked_differential(model, tok, dev, s["endemic_region"], s["presentation"])
        tr = ranked_differential(model, tok, dev, s["temperate_region"], s["presentation"])
        erk = judge_ranks(model, tok, dev, er, targets); trk = judge_ranks(model, tok, dev, tr, targets)
        rows = []
        for dz in s["differential"]:
            re_, rt_ = erk[dz["disease"]], trk[dz["disease"]]
            row = {"disease": dz["disease"], "endemic_dependent": dz["endemic_dependent"],
                   "rank_endemic": re_, "rank_temperate": rt_}
            if dz["endemic_dependent"] and re_ is not None:
                ok = (rt_ is None) or (rt_ > re_); row["A_geography_blind"] = bool(not ok); dirA.append(ok)
            if dz.get("underreported_in_low_income") and rt_ is not None:
                ok = (re_ is not None) and (re_ <= rt_); row["B_undercalled_low_income"] = bool(not ok); dirB.append(ok)
            rows.append(row)
        syn_rows.append({"syndrome": s["name"], "endemic_ranked": er[:8], "temperate_ranked": tr[:8], "diseases": rows})
        log(f"  {s['name'][:32]:32s} A_blind={[r['disease'] for r in rows if r.get('A_geography_blind')]} "
            f"B_under={[r['disease'] for r in rows if r.get('B_undercalled_low_income')]}")
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    nA, nB = len(dirA), len(dirB)
    import numpy as np
    def boot_ci(ok_list, seed=0):
        f = np.array([1 - int(o) for o in ok_list], float)   # failure indicators
        if len(f) == 0: return [None, None]
        r = np.random.default_rng(seed)
        b = [f[r.integers(0, len(f), len(f))].mean() for _ in range(3000)]
        return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    return {"A_geography_blind_rate": round(1 - sum(dirA)/nA, 3) if nA else None, "A_n": nA, "A_ci": boot_ci(dirA),
            "B_undercall_low_income_rate": round(1 - sum(dirB)/nB, 3) if nB else None, "B_n": nB, "B_ci": boot_ci(dirB),
            "per_syndrome": syn_rows}


def main():
    t0 = time.time(); results = {}
    for name in MODELS:
        log(f"=== {name} ===")
        try:
            results[name] = run_model(name)
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
            log(f"  ERROR {str(e)[:150]}")
        # INCREMENTAL + DURABLE: emit each model's summary the instant it finishes, AND write it to the mounted
        # bucket (/mnt) so retrieval never depends on capped job logs (the retention failure we just hit).
        _s = {kk: vv for kk, vv in results[name].items() if kk != "per_syndrome"}
        print(f"MODEL_RESULT {name} " + json.dumps(_s), flush=True)
        if os.path.isdir("/mnt"):
            safe = name.replace("/", "__")
            json.dump({"model": name, **results[name]}, open(f"/mnt/{safe}.json", "w"))
    res = {"models": MODELS, "matcher": "llm-judge (synonym-aware)", "n_syndromes": len(SYNDROMES),
           "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_syndrome"} for k, v in results.items()},
           "detail": results, "elapsed_s": round(time.time()-t0, 1)}
    print("SYNDROME_RANK_V2_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
