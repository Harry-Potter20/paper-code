---
title: "Guideline-grounded syndrome generator — validated pipeline"
date: 2026-07-13
premise VALIDATED: LLM-drafted differentials carry the geographic bias (my drafts dropped FSGS/hypertensive-HD/
  DCM-mis-tagged); paperclip/literature grounding RECOVERS them (FSGS, HIVAN/APOL1, glomerular-prevalence-in-
  Africa meta-analysis). So the generator MUST ground in literature, and doing so demonstrably fixes the bias.
---

# Why grounded (not LLM-drafted)
Demonstrated this session: the assistant's own LLM drafts reproduced the bias (missed FSGS/APOL1 as the top
nephrotic cause in W Africa; mis-tagged DCM/stroke). A paperclip search for "nephrotic syndrome differential
sub-Saharan Africa" immediately surfaced FSGS, HIVAN/APOL1, and a systematic review of primary glomerular-
disease PREVALENCE in Africa. => literature encodes real regional epidemiology; LLM encodes the biased average.

# Pipeline (per chief complaint)
1. TAXONOMY: curated chief-complaint list (~50-150 presenting syndromes; enumerable from ICPC-2 / standard
   presentations). Human-curated, not generated.
2. GROUND (paperclip): `search -s pmc "<complaint> differential diagnosis <region> epidemiology prevalence"`
   -> prefer systematic reviews / meta-analyses (e.g. "Prevalence of primary glomerular diseases in Africa").
   `map` to extract the differential list + region-specific prevalence per disease.
3. TAG (four-way, evidence-seeded): from the grounding evidence + WHO endemic lists + GBD:
   - endemic_dependent (exposure-required tropical; should DROP in temperate no-exposure) [Direction A]
   - low_income_enriched (genuinely MORE prevalent in low-income; excluded from bias)
   - underreported_in_low_income (genuinely under-considered treatable: autoimmune/cancer/NCD) [Direction B]
   - cosmopolitan_stable
4. MO BATCH VALIDATE: clinician approves/adjusts tags (calibration set = the standard for "correct"). Load-bearing.
5. REGION framings: disease-appropriate endemic region(s) + temperate control(s); cross for more checks.

# Output -> engine
Feeds syndrome_ranking_v2_job.py (already handles the four-way schema + LLM-judge). Scale -> bootstrap CIs +
scaling-slope p-value on the open-model law (no API needed). Frontier comparison stays manual (API-blocked).

# Validation evidence (this session)
- LLM draft nephrotic: missed FSGS, HBV-membranous, quartan malaria (MO added).
- paperclip nephrotic-Africa: recovered FSGS (PMC11561983, PMC6451079), HIVAN/APOL1 (PMC5722038), glomerular-
  prevalence meta-analysis (PMC10589414). => grounding fixes the bias in construction.

# Status
Pipeline validated + specified. Next: run steps 1-4 to expand the 13-syndrome calibration set toward ~50-100
(MO batch-validates each grounded draft), then the significance run.
