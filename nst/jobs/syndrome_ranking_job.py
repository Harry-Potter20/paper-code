# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.44", "accelerate", "huggingface_hub>=0.34"]
# ///
"""Syndrome-ranking metamorphic engine — the clinically-honest version. HF Job (t4-small).

A nonspecific syndrome (e.g. fever + thrombocytopenia) has an OVERLAPPING differential. The correct clinical
behaviour is that EXPOSURE-DEPENDENT (endemic) diseases should be RANKED lower / drop out when the patient is
in a non-endemic setting with no exposure, while COSMOPOLITAN diseases stay. We present each syndrome in an
endemic vs temperate framing, get the model's RANKED differential, and score the rank-shift RELATIONALLY
against one clinical tag per disease (endemic_dependent). No per-case ground truth.

Metrics (over endemic_dependent diseases the model lists in the endemic framing):
  - appropriate_downweight: disease drops out OR moves DOWN the ranking in the temperate framing (GOOD).
  - geography_blind: disease stays at same-or-better rank in temperate despite no exposure (the FAILURE).
Also: does the model PROMOTE cosmopolitan diseases in the temperate framing (sane behaviour)?
Emits SYNDROME_RANK_RESULT.
"""
from __future__ import annotations
import json, os, time, re
def log(m): print(f"[synd] {m}", flush=True)

MODEL = os.environ.get("AUDIT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SYNDROMES = json.loads(os.environ["SYNDROMES_JSON"])["syndromes"]

PROMPT = """You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.
Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}
Setting: presenting at {region}.
Case: {presentation}.
JSON:"""


def parse_list(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0), strict=False)
            if isinstance(d.get("ranked_differential"), list):
                return [str(x).lower() for x in d["ranked_differential"]]
        except Exception:
            pass
    return []


def rank_of(keywords, ranked):
    for i, item in enumerate(ranked):
        if any(k.lower() in item for k in keywords):
            return i
    return None


def get_ranked(model, tok, dev, region, presentation):
    import torch
    msgs = [{"role": "user", "content": PROMPT.format(region=region, presentation=presentation)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
    return parse_list(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


def main():
    t0 = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 device_map=dev).eval()
    log(f"loaded {MODEL} (t={time.time()-t0:.0f}s)")

    syn_rows = []; dirA = []; dirB = []
    for s in SYNDROMES:
        er = get_ranked(model, tok, dev, s["endemic_region"], s["presentation"])
        tr = get_ranked(model, tok, dev, s["temperate_region"], s["presentation"])
        dz_rows = []
        for dz in s["differential"]:
            re_ = rank_of(dz["keywords"], er); rt_ = rank_of(dz["keywords"], tr)
            row = {"disease": dz["disease"], "endemic_dependent": dz["endemic_dependent"],
                   "underreported_in_low_income": dz.get("underreported_in_low_income", False),
                   "rank_endemic": re_, "rank_temperate": rt_}
            # Direction A: exposure-dependent disease should DROP/down-rank where there is no exposure
            if dz["endemic_dependent"] and re_ is not None:
                ok = (rt_ is None) or (rt_ > re_)
                row["A_appropriate_downweight"] = bool(ok); row["A_geography_blind"] = bool(not ok)
                dirA.append(ok)
            # Direction B: cosmopolitan disease must NOT be dropped/down-ranked in the low-income framing
            if (not dz["endemic_dependent"]) and rt_ is not None:
                ok = (re_ is not None) and (re_ <= rt_)     # present in endemic at same-or-better rank
                row["B_appropriate_retained"] = bool(ok); row["B_undercalled_low_income"] = bool(not ok)
                dirB.append(ok)
            dz_rows.append(row)
        syn_rows.append({"syndrome": s["name"], "endemic_ranked": er[:8], "temperate_ranked": tr[:8],
                         "diseases": dz_rows})
        ablind = [r["disease"] for r in dz_rows if r.get("A_geography_blind")]
        bunder = [r["disease"] for r in dz_rows if r.get("B_undercalled_low_income")]
        log(f"{s['name'][:34]:34s} A_over-call={ablind} | B_under-call={bunder}")

    nA, nB = len(dirA), len(dirB)
    summary = {"n_syndromes": len(SYNDROMES),
               "A_geography_blind_rate": round(1 - sum(dirA) / nA, 3) if nA else None,   # over-call outside
               "A_n_checks": nA,
               "B_undercall_low_income_rate": round(1 - sum(dirB) / nB, 3) if nB else None,  # under-call inside
               "B_n_checks": nB}
    res = {"model": MODEL, "method": "syndrome-ranking metamorphic (rank-shift, tag-based, vignette-independent)",
           "summary": summary, "per_syndrome": syn_rows,
           "verdict": {"note": ("BIDIRECTIONAL. Direction A (A_geography_blind_rate): exposure-dependent disease that "
                                "FAILED to drop where there is no exposure (over-calling tropical disease outside endemic "
                                "areas). Direction B (B_undercall_low_income_rate): cosmopolitan/autoimmune/NCD disease "
                                "that got DROPPED or down-ranked in the low-income framing (under-calling e.g. SLE/lymphoma "
                                "inside low-income settings — the 'it's-all-infection-in-Africa' bias, driven by under-"
                                "reporting = training-data rarity). Only clinical input = the tags; no per-case authoring.")},
           "elapsed_s": round(time.time() - t0, 1)}
    print("SYNDROME_RANK_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
