# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.50,<5", "accelerate", "numpy", "huggingface_hub>=0.34", "sentencepiece", "protobuf"]
# ///
"""No Safe Tier R3 (NST_R3_ROBUSTNESS_PREREG_2026-11-10.md): unified generation pass. Generation ONLY; scoring is done
offline by r3_analyze.py with the paper's deterministic matcher.
Per model: 26 validated syndromes x 4 settings (x1 factorial; two cells are the paper's exact framings) x 3 templates,
greedy, bf16, batched (left padding). Plus T1 sampling (T 0.7, top-p 0.95, 5 seeds) on the paper's 2 settings for
models listed in SAMPLE_MODELS.
Output: /mnt/processed/nst/r3/<model>.json = {model, repo_used, raw: {template: {setting: {syndrome: [list]}}},
  sampled: {seed: {setting: {syndrome: [list]}}}, malformed: {...}}. Skips models whose output already exists.
Env: MODELS (comma list), SAMPLE_MODELS, LIMIT (smoke: first N syndromes), BATCH (16), OUT_DIR.
"""
from __future__ import annotations
import gc, json, os, re, time
import numpy as np

X6 = "/mnt/processed/nst/x6"
SYN = json.load(open(f"{X6}/syndromes_26.json"))["syndromes"]
SET = {s["name"]: s["settings"] for s in json.load(open(f"{X6}/x1_resource_settings_draft.json"))["syndromes"]}
if os.environ.get("LIMIT"):
    SYN = SYN[:int(os.environ["LIMIT"])]
for s in SYN:                                   # the two paper cells must be the paper's exact text
    assert SET[s["name"]]["low_income__constrained"] == s["endemic_region"], s["name"]
    assert SET[s["name"]]["high_income__well_resourced"] == s["temperate_region"], s["name"]
CELLS = ["low_income__constrained", "high_income__well_resourced", "low_income__well_resourced", "high_income__constrained"]
PAPER = CELLS[:2]
MODELS = [m.strip() for m in os.environ.get("MODELS", "").split(",") if m.strip()]
SAMPLE = {m.strip() for m in os.environ.get("SAMPLE_MODELS", "Qwen/Qwen2.5-7B-Instruct,m42-health/Llama3-Med42-8B").split(",")}
MIRROR = {"meta-llama/Meta-Llama-3-8B-Instruct": "NousResearch/Meta-Llama-3-8B-Instruct",       # pre-declared exception
          "google/gemma-3-4b-it": "unsloth/gemma-3-4b-it"}                                     # amendment A1 (gated)
BATCH = int(os.environ.get("BATCH", "16"))
OUT = os.environ.get("OUT_DIR", "/mnt/processed/nst/r3" + ("_smoke" if os.environ.get("LIMIT") else "")); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
JSONSPEC = 'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
TEMPLATES = {
    "T1": ('You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.\n' + JSONSPEC +
           'Setting: {region}\nCase: {presentation}.\nJSON:'),
    "T2": ('You are an experienced physician. List the differential diagnoses for this patient in order of likelihood, '
           'most likely first.\nPatient: {presentation}.\nWhere the patient is being seen: {region}.\n' + JSONSPEC + 'JSON:'),
    "T3": ('Patient: {presentation}.\nLocation and facility: {region}.\nWhat are the possible diagnoses, most likely '
           'first?\n' + JSONSPEC + 'JSON:'),
}
MISTRAL_V01_TEMPLATE = ("{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
                        "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% else %}{{ message['content'] + eos_token }}"
                        "{% endif %}{% endfor %}")


def log(m):
    print(f"[r3 {time.time() - T0:6.0f}s] {m}", flush=True)


def parse(txt):
    """JSON first; if absent/invalid, fall back to numbered or bulleted list lines (amendment A1, applied to ALL models).
    Returns (list, malformed_json, used_fallback)."""
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        rl = json.loads(m.group(0), strict=False).get("ranked_differential", []) if m else None
    except Exception:
        rl = None
    if isinstance(rl, list):
        return [str(x) for x in rl], False, False
    t = re.search(r'"ranked_differential"\s*:\s*\[(.*)', txt, re.S)              # truncated JSON (never closed)
    if t:
        items = re.findall(r'"([^"]+)"', t.group(1))
        if items:
            return items, True, True
    items = []
    for line in txt.splitlines():
        mm = re.match(r"\s*(?:\d+[.)]|[-*\u2022])\s+(.+)", line)
        if mm:
            it = re.split(r"\s+[-\u2013:(]\s*", mm.group(1).strip().strip("*"), maxsplit=1)[0].strip(" .*\"'")
            if it:
                items.append(it)
    return items[:15], True, bool(items)


def load(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = name
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        if name not in MIRROR:
            raise
        log(f"{name} not loadable ({type(e).__name__}); pre-declared mirror {MIRROR[name]}"); repo = MIRROR[name]
        tok = AutoTokenizer.from_pretrained(repo)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = MISTRAL_V01_TEMPLATE                                  # BioMistral ships no template
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    return repo, tok, model


def generate(tok, model, prompts, sample=False, seed=0):
    import torch
    outs = []
    for k in range(0, len(prompts), BATCH):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[k:k + BATCH]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=300, pad_token_id=tok.pad_token_id)
        if sample:
            torch.manual_seed(seed + k); kw.update(do_sample=True, temperature=0.7, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            o = model.generate(**enc, **kw)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def run(name):
    import torch
    repo, tok, model = load(name)
    raw, mal, sampled, snips, jok = {}, {}, {}, {}, {}
    for tn, tpl in TEMPLATES.items():
        keys = [(c, s["name"]) for c in CELLS for s in SYN]
        prompts = [tpl.format(region=SET[n][c], presentation=next(s for s in SYN if s["name"] == n)["presentation"]) for c, n in keys]
        texts = generate(tok, model, prompts); raw[tn] = {c: {} for c in CELLS}; mal[tn] = {"json_malformed": 0, "fallback_used": 0, "empty": 0}
        for (c, n), t in zip(keys, texts):
            rl, bad, fb = parse(t); raw[tn][c][n] = rl; jok.setdefault(tn, {}).setdefault(c, {})[n] = not bad
            mal[tn]["json_malformed"] += int(bad); mal[tn]["fallback_used"] += int(fb); mal[tn]["empty"] += int(not rl)
            if bad and len(snips.setdefault(tn, [])) < 6:
                snips[tn].append(t[:400])
        log(f"{name} {tn}: {len(texts)} gens, {mal[tn]}, mean len {np.mean([len(v) for c in CELLS for v in raw[tn][c].values()]):.2f}")
    if name in SAMPLE:
        keys = [(c, s["name"]) for c in PAPER for s in SYN]
        prompts = [TEMPLATES["T1"].format(region=SET[n][c], presentation=next(s for s in SYN if s["name"] == n)["presentation"]) for c, n in keys]
        for seed in range(5):
            texts = generate(tok, model, prompts, sample=True, seed=1000 * seed); sampled[seed] = {c: {} for c in PAPER}
            for (c, n), t in zip(keys, texts):
                sampled[seed][c][n] = parse(t)[0]
        log(f"{name}: 5 sampled runs done")
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"model": name, "repo_used": repo, "raw": raw, "malformed": mal, "sampled": sampled, "malformed_snippets": snips, "json_ok": jok}


def main():
    for name in MODELS:
        path = f"{OUT}/{name.replace('/', '__')}.json"
        if os.path.exists(path):
            log(f"skip {name} (exists)"); continue
        log(f"=== {name} ===")
        try:
            r = run(name)
        except Exception as e:                                                   # report, never substitute
            r = {"model": name, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            log(f"ERROR {name}: {r['error']}")
            json.dump(r, open(path + ".error", "w"))                             # never blocks a rerun
        else:
            json.dump(r, open(path, "w"))
        print("R3_MODEL_DONE " + json.dumps({"model": name, "repo_used": r.get("repo_used"), "malformed": r.get("malformed"),
                                             "error": r.get("error")}), flush=True)
    print("R3_DONE", flush=True)


if __name__ == "__main__":
    main()
