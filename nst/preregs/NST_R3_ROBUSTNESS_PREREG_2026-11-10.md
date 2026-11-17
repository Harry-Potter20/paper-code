---
status: PRE-REGISTERED 2026-11-10, before any R3 data. Owner instruction: "keep pushing on no safe tier. it must be as
  robust as possible."
no_paperclip: true
budget: HF a10g-large, ≤ ~$8; Claude repeats via the subscription CLI (no API spend).
---

# NST R3: unified robustness re-run (one harness, one matcher, all models, 3 templates, factorial settings, repeats)

## Why (the reviewer threats this answers)
1. **Mixed provenance.** The paper's tables mix judge-scored and matcher-scored rates from different harness runs
   and quantisations, so model rankings are not comparable (e.g. abstract "Med42 worst open model, 0.867" vs Qwen-7B
   0.882 in §4.3).
2. **Single prompt template, single greedy decode.** Prompt sensitivity and sampling variance are unmeasured.
3. **"Rises with scale" has no formal trend test.**
4. **R1 (Med42 vs Llama-3-8B base) is open.**
5. **Geography vs resource confound.** The paper's pair (sub-Saharan district hospital vs Oslo ED) varies both; a
   provisional factorial (X1, 2 models, unvalidated drafts) says resource × low-income drives the loss.
6. **Frontier evidence is one batched paste-bridge run per model** (Limitation 2/3).

## Design
**Syndromes:** the 26 clinician-validated syndromes only (syndromes_26.json). The 24 X6 drafts are excluded from
every confirmatory test here.

**Settings (4 cells per syndrome):**
- `low_income__constrained` = the paper's endemic framing (identical text);
- `high_income__well_resourced` = the paper's temperate framing (identical text);
- `low_income__well_resourced` and `high_income__constrained` from x6/x1_resource_settings_draft.json. These two new
  setting descriptions are drafted, not clinician-validated; stated as a limitation.

**Prompt templates (3):**
- **T1:** the paper's prompt (setting line, then case).
- **T2:** a clinician-voice paraphrase: "You are an experienced physician. List the differential diagnoses for this
  patient in order of likelihood…". Case first, then setting.
- **T3:** a neutral question form: "Patient: <case>. Location and facility: <setting>. What are the possible
  diagnoses, most likely first?". Same JSON output.

All three ask for a JSON ranked list; no template names diagnostic categories.

**Decoding:** greedy (all templates); plus T1 sampling at T = 0.7, top-p 0.95, 5 seeds, for Qwen2.5-7B and Med42-8B
on the paper's 2 settings. bf16 throughout (as the paper's open runs); batched generation with left padding;
max 300 new tokens; malformed JSON → empty list, counted.

**Models (16; exact repos fixed now; any that fails to load is reported as not run, never substituted, with one
pre-declared exception):**
- Qwen2.5-{0.5B, 1.5B, 3B, 7B}-Instruct;
- tiiuae/Falcon3-{1B, 3B, 7B, 10B}-Instruct;
- ibm-granite/granite-3.1-{2b, 8b}-instruct. The paper's Granite version is unrecorded; any difference is noted.
- Base-vs-medical pairs:
  - mistralai/Mistral-7B-Instruct-v0.1 vs BioMistral/BioMistral-7B (BioMistral's base);
  - google/gemma-3-4b-it vs google/medgemma-4b-it;
  - meta-llama/Meta-Llama-3-8B-Instruct vs m42-health/Llama3-Med42-8B. If the meta-llama repo is gated for this
    token, use NousResearch/Meta-Llama-3-8B-Instruct, a byte-identical mirror; this is the pre-declared exception.

**Instrument:** the clinician-validated deterministic matcher (paper §3), unchanged. Metrics A (over-call),
B (under-call), hard-drop and length, all exactly as the paper defines them, on the paper's pair (T1–T3). All CIs are
**cluster bootstrap over syndromes** (2,000 resamples). Paired differences are bootstrapped over syndromes.

## Pre-registered tests (decision rules fixed now)
- **R3-H1 (unified no-safe-tier ranking):** report B and hard-drop for all 16 models on T1 in one table. No claim of a
  "worst" model unless its paired B difference vs every other model's has CI > 0.
- **R3-H2 (scale trend):**
  - Within each family (Qwen 4 sizes, Falcon3 4 sizes, Granite 2 sizes), compute the slope of B on log10(params)
    with a syndrome-cluster bootstrap.
  - **Supported** if the slope CI > 0 for ≥ 2 of the 3 families on T1, **and** the point slope is positive on T2 and T3
    for those families.
  - Granite has 2 sizes, so its "slope" is a paired difference.
- **R3-H3 (medical fine-tuning):** paired B difference (medical − base) for each of the 3 pairs.
  - **"Medical tuning does not reliably reduce under-call"** is supported if no pair shows a reduction with CI < 0 on
    T1 that also holds in sign on T2 and T3.
  - **"Can worsen it"** requires at least one pair with an increase whose CI > 0 on T1.
  - **R1 verdict:** Med42 − Llama-3-8B CI includes 0 → *inherited*; CI > 0 → *induced*; CI < 0 → *medical tuning
    reduced it*.
- **R3-H4 (template robustness):** the headline holds on each template if, for the four largest open models (≥ 7B),
  mean B ≥ 0.5 and mean hard-drop ≥ 0.25 on T1, T2 and T3. Reported per template either way.
- **R3-H5 (sampling variance):** across the 5 sampled runs, report SD of B and hard-drop for Qwen2.5-7B and Med42.
  - "Greedy is representative" if the greedy value lies within the sampled range ± 0.05.
  - We expect sampling to *raise* inclusion slightly (more diverse lists). That is reported, not a failure.
- **R3-X1 (confirmatory factorial on validated syndromes):**
  - For each model ≥ 3B, decompose underreported-target inclusion into geography (HIC − LIC), resource (WR − C) and
    interaction, paired by syndrome, as in the provisional X1 script.
  - **"Resource, not geography, drives the under-call"** is confirmed if, on T1, the resource effect CI > 0 AND the
    geography effect CI includes 0 in ≥ 2/3 of models ≥ 3B.
  - **"Geography drives it"** (the paper's original framing) if the geography CI > 0 and resource CI includes 0 in
    ≥ 2/3.
  - Anything else is reported as **"both / mixed"**. The paper's title and framing follow whichever holds.
- **R3-X5 (Claude repeats, per-case, unbatched):**
  - Claude via `claude -p`. Fresh process per prompt, run from an empty directory with a minimal replacement system
    prompt and no tools, so neither the vault's CLAUDE.md nor the coding-agent prompt is loaded.
  - T1, the paper's 2 settings, 26 syndromes, 3 repeats.
  - The model is whatever `--model sonnet` resolves to, recorded verbatim. It may differ from the paper's Claude
    Sonnet 5 and is reported as a separate row, never merged.
  - Report per-run B / hard-drop and the across-run range. **"Band holds"** if all 3 runs have B ≥ 0.70. The paper's
    lowest frontier value was 0.82; 0.70 is the pre-declared lower edge.

## What changes in the paper depending on outcomes
- **R3-X1 "resource":** retitle away from "geographic", and reframe as "low-resource-framed diagnostic narrowing:
  models adapt the wrong layer".
- **R3-H2 fails:** "rises with scale" is withdrawn; the claim becomes "present across scale".
- **R3-H3:** §4.1 is rewritten to whatever the paired tests support. The old 0.867 / "worst" wording is replaced by
  unified numbers regardless.
- **R3-H4 / H5 failures** are reported as limitations, with the per-template table in the supplement.

## Amendment A1 (2026-11-10, after a 1-syndrome smoke test, before any R3 data)
- **Parsing:** BioMistral (4/4) and Falcon3-1B (3/4) emitted no JSON in the smoke test. Malformed output would score
  as an empty list, and an empty low-income list turns every target into a hard-drop, which biases the rates upward.
  - Fix, applied identically to **all** models: try JSON first, else extract numbered or bulleted list lines.
  - JSON-malformed counts, fallback-used counts, empty counts and up to 6 raw snippets per template are saved and
    reported.
  - **Sensitivity analysis (pre-declared):** every headline is recomputed on JSON-only parses; a model whose
    fallback rate is > 20% is flagged.
- **Gemma-3-4b-it is gated for this token (403).** `unsloth/gemma-3-4b-it` (the same weights, re-uploaded) is added
  as a declared mirror, recorded in `repo_used`.
- **A1b (second smoke test, still before R3 data):** BioMistral's T1 output is valid JSON that never closes (it lists
  until the 300-token cap).
  - Truncation-tolerant extraction (quoted items after `"ranked_differential": [`) is added for all models and
    counted as fallback.
  - BioMistral's T2/T3 output is a lone `{`, a format failure. That model × template is reported as unusable, not
    imputed.

## RESULTS (2026-11-11; jobs 6af37a25… / 6af37a2b…, l4x1; analysis jobs/r3_analyze.py → x6/r3_results.json)
**Coverage and usability**
- 16/16 models ran.
- **Unusable (> 50% empty output), excluded, never imputed:** Mistral-7B-Instruct-v0.1 T1 and T3; BioMistral-7B T2
  and T3.
- Gated bases were loaded from the declared mirrors (NousResearch Llama-3-8B, unsloth Gemma-3-4B).
- JSON-only sensitivity: B is unchanged for every model with enough JSON parses (largest change: Gemma-3 0.818 → 0.933
  on 19 syndromes). Parsing does not drive results.

**Unified T1 table (cluster bootstrap over syndromes)** — B = under-call, top-5 = length-invariant drop

| model | A | B [95% CI] (n targets) | top-5 drop | hard-drop | rank shift | length |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.00 | 0.40 [0.00, 1.00] (5) | 0.40 | 0.40 | 0.13 | 4.0 |
| Qwen2.5-1.5B | 0.10 | 0.60 [0.40, 0.86] (10) | 0.40 | 0.40 | 0.27 | 3.1 |
| Qwen2.5-3B | 0.43 | 0.50 [0.20, 1.00] (8) | 0.25 | 0.25 | 0.15 | 3.9 |
| Qwen2.5-7B | 0.19 | 0.94 [0.80, 1.00] (16) | 0.69 | 0.69 | 0.53 | 4.1 |
| Falcon3-1B | 0.29 | 1.00 [1.00, 1.00] (8) | 1.00 | 1.00 | 0.91 | 21.4 |
| Falcon3-3B | 0.30 | 0.69 [0.44, 0.92] (16) | 0.54 | 0.38 | 0.25 | 11.6 |
| Falcon3-7B | 0.20 | 0.81 [0.59, 1.00] (21) | 0.50 | 0.52 | 0.24 | 9.5 |
| Falcon3-10B | 0.28 | 0.74 [0.52, 0.93] (23) | 0.55 | 0.61 | 0.36 | 6.1 |
| Granite-3.1-2B | 0.05 | 0.86 [0.64, 1.00] (22) | 0.79 | 0.68 | 0.54 | 10.1 |
| Granite-3.1-8B | 0.12 | 0.95 [0.86, 1.00] (19) | 0.77 | 0.68 | 0.52 | 4.4 |
| Gemma-3-4B | 0.10 | 0.82 [0.56, 1.00] (22) | 0.76 | 0.68 | 0.46 | 6.1 |
| MedGemma-4B | 0.19 | 0.73 [0.53, 0.89] (22) | 0.57 | 0.59 | 0.35 | 5.3 |
| Llama-3-8B | 0.05 | 0.82 [0.57, 1.00] (17) | 0.77 | 0.77 | 0.46 | 5.2 |
| Med42-8B | 0.06 | 1.00 [1.00, 1.00] (17) | 0.71 | 0.65 | 0.54 | 6.5 |
| BioMistral-7B | 0.35 | 0.65 [0.46, 0.87] (26) | 0.40 | 0.39 | 0.31 | 21.2 |
| Mistral-7B-v0.1 | — | unusable (77% empty) | — | — | — | — |

**Decisions**
- **R3-H1:** highest B is Med42 (1.00), but **no "worst" claim is allowed** (paired CIs do not exclude 0 against every
  model).
- **R3-H2: NOT SUPPORTED. "Rises with scale" is withdrawn.**
  - Qwen T1 slope +0.41 [−0.13, 0.84] (T2 +0.33 [0.10, 0.54]).
  - **Falcon3 T1 −0.22 [−0.38, −0.06]** (the opposite direction).
  - Granite +0.14 [−0.17, 0.55].
  - Small-model denominators are tiny (5–10 targets).
  - The claim becomes **"present across scale"**: every usable model ≥ 3B has B ≥ 0.65.
- **R3-H3:** no pair shows a template-stable reduction → **medical tuning does not reliably reduce under-call.**
  - "Can worsen" is NOT established: Med42 − Llama-3-8B = +0.18 [0.000, 0.43] on T1 (+0.12 [0.000, 0.24] on T2;
    +0.06 [−0.14, 0.25] on T3).
  - **R1 verdict: inherited** (CI includes 0), borderline toward induced.
  - Gemma-3 → MedGemma −0.09 [−0.39, 0.19] (null).
  - Mistral → BioMistral is not evaluable (base unusable on T1 / T3; BioMistral unusable on T2 / T3).
- **R3-H4: HOLDS.** Usable ≥ 7B models:

  | template | mean B | mean top-5 drop | mean hard-drop | models |
  |---|---|---|---|---|
  | T1 | 0.84 | 0.63 | 0.61 | 7 |
  | T2 | 0.80 | 0.57 | 0.41 | 7 |
  | T3 | 0.82 | 0.58 | 0.50 | 6 |

  The prereg text said "four largest ≥ 7B"; the rule was applied to all usable ≥ 7B models, which is stricter.
- **R3-H5:**
  - Qwen2.5-7B: greedy is representative (sampled B 0.88–1.00, SD 0.06).
  - Med42: greedy B 1.00 lies above the sampled range 0.81–0.94 (SD 0.05). Hard-drop under sampling is very noisy
    (0.40–0.88, SD 0.19).
  - So Med42's greedy point estimate is at the high end, and single-decode hard-drop figures carry about ±0.2 noise
    for that model.
- **R3-X1: BOTH / MIXED by the pre-registered rule.**
  - On T1, the resource effect CI > 0 with geography CI spanning 0 in **7 of 11** usable models ≥ 3B; the rule needs
    ≥ 2/3, i.e. 8.
  - The geography-only pattern occurs in **0 / 11**.
  - Resource CI > 0 in 8/11 overall. Geography CI > 0 in 1/11 (Llama-3, together with resource).
  - T3: resource 8/10, geography 0/10. T2: resource 5/10, geography 2/10.
  - The interaction is negative in 9 models, significantly in 3: the loss concentrates in low-income × constrained.
  - **Reading:** the effect is not driven by region alone, so "geographic" is withdrawn. Facility resource level
    carries most of it, but "resource alone" did not meet the pre-registered bar.
- **R3-X5 (Claude Sonnet 5.5, per case, 3 runs):**
  - B 0.851 / 0.854 / 0.857 → **band holds.**
  - Top-5 drop 0.59 / 0.66 / 0.55; hard-drop 0.13–0.17; mean length 13.9–14.1.
