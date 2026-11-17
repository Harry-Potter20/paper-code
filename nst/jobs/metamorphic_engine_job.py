# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.44", "accelerate", "huggingface_hub>=0.34"]
# ///
"""Metamorphic geography engine — VIGNETTE-INDEPENDENT endemic-diagnosis audit. HF Job (t4-small).

Auto-generates matched endemic/temperate framings from a public disease TABLE (no hand-authored vignettes),
and scores the metamorphic relation RELATIONALLY (no per-case clinical ground truth):
  - geography_sensitive: disease appears in the ENDEMIC differential but DROPS in the temperate (no-exposure)
    framing  -> the model uses local epidemiology.
  - ignores_geography: disease appears in BOTH -> the model anchors on the lab/symptom pattern and ignores
    where the patient is (the flagship failure).
  - endemic_misattribution: in the ENDEMIC framing, a TEMPERATE-confuser is the primary dx -> model chases
    the wrong, region-inappropriate disease.
Same structured-JSON prompt as the audit. Scales to any number of diseases/regions. Emits METAMORPH_RESULT.
"""
from __future__ import annotations
import json, os, time, re
def log(m): print(f"[metamorph] {m}", flush=True)

MODEL = os.environ.get("AUDIT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DISEASES = json.loads(os.environ["DISEASES_JSON"])["diseases"]

PROMPT = """You are assisting a clinician at the point of care. Read the case and respond with ONLY a JSON object:
{{"primary_diagnosis": "<one>", "differential": ["<dx>", ...], "triage": "<EMERGENCY|ADMIT|TREAT_OBSERVE|OUTPATIENT>"}}
Setting: presenting at {region}.
Case: {presentation}.
JSON:"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0), strict=False)
    except Exception:
        return {}


def run(model, tok, dev, region, presentation):
    import torch
    msgs = [{"role": "user", "content": PROMPT.format(region=region, presentation=presentation)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    p = parse_json(txt)
    primary = str(p.get("primary_diagnosis", "")).lower()
    blob = (primary + " " + " ".join(map(str, p.get("differential", [])))).lower()
    return primary, blob, str(p.get("triage", "")).upper()


def any_in(blob, keys): return any(k.lower() in blob for k in keys)


def main():
    t0 = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 device_map=dev).eval()
    log(f"loaded {MODEL} (t={time.time()-t0:.0f}s)")

    rows = []
    for d in DISEASES:
        ep, eb, etr = run(model, tok, dev, d["endemic_region"], d["presentation"])
        tp, tb, ttr = run(model, tok, dev, d["temperate_region"], d["presentation"])
        appears_e = any_in(eb, d["keywords"]); appears_t = any_in(tb, d["keywords"])
        misattr = any_in(ep, d.get("temperate_confusers", []))     # confuser is PRIMARY in endemic framing
        rows.append({
            "disease": d["name"], "endemic_primary": ep[:60], "temperate_primary": tp[:60],
            "appears_endemic": appears_e, "appears_temperate": appears_t,
            "geography_sensitive": bool(appears_e and not appears_t),
            "ignores_geography": bool(appears_e and appears_t),
            "missed_in_endemic": bool(not appears_e),
            "endemic_misattribution": bool(misattr)})
        log(f"{d['name']:22s} endemic_primary='{ep[:35]}' | appears E/T={appears_e}/{appears_t} "
            f"geo_sensitive={rows[-1]['geography_sensitive']} misattr={misattr}")

    n = len(rows)
    def rate(k): return round(sum(r[k] for r in rows) / n, 3)
    summary = {
        "n_diseases": n,
        "appears_endemic_rate": rate("appears_endemic"),           # did it even consider the right disease?
        "geography_sensitive_rate": rate("geography_sensitive"),   # correctly localizes
        "ignores_geography_rate": rate("ignores_geography"),       # flagship failure
        "missed_in_endemic_rate": rate("missed_in_endemic"),       # didn't consider endemic dx at all
        "endemic_misattribution_rate": rate("endemic_misattribution")}
    res = {"model": MODEL, "method": "metamorphic geography (vignette-independent; relational scoring)",
           "summary": summary, "per_disease": rows,
           "verdict": {
               "note": ("ignores_geography_rate high => model anchors on symptom/lab pattern and does not use local "
                        "epidemiology (flagship failure). geography_sensitive_rate = fraction it correctly localizes. "
                        "No per-case clinical ground truth used; disease TABLE is MO-validatable public facts.")},
           "elapsed_s": round(time.time()-t0, 1)}
    print("METAMORPH_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
