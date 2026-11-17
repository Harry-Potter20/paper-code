---
status: PLAN (owner go to expand, 2026-11-10). Nothing run. No Paperclip anywhere in this paper (agreement: MMBU only).
supersedes: the 2026-11-08 revision list in PAPERS_LEDGER (folds it in).
---

# No Safe Tier: expansion to answer the resource-adaptation objection

## The objection, stated at full strength
"Low-resource medicine runs on improvisation: you cannot order a biopsy, an MRI or a flow panel, so a clinician in
a district hospital rationally narrows to what is common and treatable with what they have. A model that does the
same is adapting to context, not biased. Holding it to a high-resource differential is itself a form of
epistemic imperialism." The second horn, raised by the owner: improvisation norms also breed harmful practice
(empiric treat-first, test-never), so a model that mirrors them propagates harm, but a model that refuses them is
"unrealistic".

## The answer: three things the current draft conflates
1. **Diagnosis is resource-invariant.** Knowing that lymphoma, lupus or FSGS belongs on the differential costs
   nothing. Resource limits constrain what you can *do* about a hypothesis, not whether you *hold* it. Dropping it
   removes the trigger for referral, a repeat visit, a cheap discriminating test, or a safety-net instruction.
2. **Management is legitimately resource-stratified.** The resource-stratified guideline literature formalises
   context-appropriate care in tiers (basic / limited / enhanced / maximal):
   - Breast Health Global Initiative;
   - NCCN Harmonized Guidelines for sub-Saharan Africa;
   - WHO PEN;
   - WHO "test before treat" for malaria.

   Adapting the *plan* to the tier is good medicine and is scored as such.
3. **Harmful improvisation is a measurable third category.** These are actions that the resource-stratified
   guidelines reject *even at the basic tier*:
   - empiric anti-TB therapy without attempting sputum/Xpert where available;
   - antimalarials after a negative RDT;
   - steroids before excluding infection;
   - treating a mass as infection without a biopsy/referral plan.

   Under-call and harmful improvisation are linked: a lymphoma that never makes the differential gets treated as TB.

The paper's claim becomes sharper and harder to dismiss: **models adapt the wrong layer.** They narrow the
*diagnosis* (which should not move) by geography, while the *management* adaptation (which should move, with
resources) is the thing to measure separately.

## New experiments (all CPU or API; no new architecture)

**X1. Geography × resources factorial: what actually drives the under-call.**
- **Design:** the current endemic vs temperate pairs gain a resource axis, 2 × 2 per syndrome:
  - {low-income region, high-income region} × {well-resourced facility, constrained facility};
  - e.g. a private tertiary hospital in Lagos with CT, biopsy and flow cytometry, vs a remote rural clinic in the
    Scottish Highlands with a 2-day courier for bloods;
  - same presentation and exposures held fixed where geography allows.
- **Metric:** under-call / hard-drop (existing matcher), two-way decomposition into a geography main effect, a
  resource main effect, and the interaction.
- **Prediction (pre-register):** the geography effect dominates; the resource effect on the *differential* is small.
  If models were "adapting to resources", the resource effect would dominate and Lagos-tertiary would look like
  London.
- **If the resource effect dominates instead,** the paper's framing changes and we say so.

**X2. Management tiering against resource-stratified guidelines.**
- **Design:** for each syndrome × resource tier, ask for next steps, not just a differential.
- **Scoring:** a per-syndrome rubric derived from the stratified guidelines, three labels per recommended action:
  - **tier-appropriate:** uses what the tier has, e.g. VIA instead of cytology, ultrasound instead of CT, clinical
    criteria, RDT;
  - **unavailable escalation:** orders what the tier lacks, without a referral framing;
  - **harmful improvisation:** guideline-contraindicated at every tier.
- **Rubric:** authored from the guideline documents (web, not Paperclip), validated by the owner plus ≥ 1
  independent clinician, κ reported.
- **Key analysis:** is harmful improvisation more frequent when the target diagnosis was under-called? That links the
  two horns quantitatively.

**X3. Prevalence control (GBD 2021) — from the 2026-11-08 plan.**
- **Design:** prevalence ratio (sub-Saharan Africa vs Western Europe) per underreported target.
- **Analysis:** under-call vs ratio; the claim holds if under-call persists for targets with ratio ≥ 1.
  `low_income_enriched` is already excluded; promote that from one Methods line to an argued section.
- **Needs:** the owner's IHME Results Tool export.

**X4. The harness fix, tested (turns an audit into audit + remedy; answers "limited innovation").**
- **Design:** the Discussion already recommends a location-blind pass; test it. Two-pass harness:
  1. a location-blind differential;
  2. location-aware *re-ranking and annotation*, where the model may reorder and add, but may not delete a pass-1
     item without stating why.
- **Variants:** also a one-line "also list treatable everywhere-causes" instruction.
- **Metrics:** under-call, hard-drop, over-call (must keep the over-call fix), differential length.
- **Pre-registered success:** hard-drop reduced ≥ 50% with over-call not worse than single-pass by > 5 pts, on open
  ladders + ≥ 1 frontier model.

**X5. Frontier variance (the outstanding Limitation 2) + automation.**
- 3–5 repeat runs on a 10-syndrome subset.
- Claude can be scripted via the subscription CLI (`claude -p`, as Popquiz does), per-case not batched, which also
  addresses Limitation 3 (batching) for one frontier model. GPT and Gemini stay on the paste-bridge for a subset.

**X6. Benchmark expansion 26 → ~50 syndromes, without Paperclip.**
- **Sources:** WHO and national guidelines, textbook differentials, and the owner's clinical case knowledge. The
  generator spec stays, but grounding moves to cited guideline text.
- **Validation:** clinician spot-check as before. Hold the new 24 out as a confirmatory set: the headline must
  replicate on syndromes authored after the original findings.

**X7. Clinician audit, enlarged:** matcher + taxonomy on ≥ 60 cases by ≥ 2 clinicians including ≥ 1 practising in
Nigeria; κ.

## Paper changes
- New §2.x "Resource adaptation vs diagnostic narrowing" (the conceptual distinction, with the stratified-guideline
  literature).
- New results §4.6 (X1), §4.7 (X2), §4.8 (X4); prevalence (X3) folds into §4.5.
- Abstract and title keep "No Safe Tier"; add the harness result if X4 passes.
- Mechanism (§5) gains the layer argument: capability sharpens the diagnostic prior (wrong layer) without adding
  management-tier awareness (right layer). That is testable in X1 + X2.
- Venue: PLOS Digital Health or npj Digital Medicine; BMJ Global Health if the X2 framing leads.

## Order and cost
1. X4 (cheap, on open models).
2. X1.
3. X3 (needs the GBD export).
4. X5 (Claude CLI, scripted).
5. X2 (rubric authoring is the slow part).
6. X6 and X7 (clinician time).

Open-model runs are small HF jobs, roughly $2–5 total. Frontier runs use the subscription CLI and the paste-bridge,
with no API spend.

## What the owner must supply
- The GBD export.
- Clinician time for the X2 rubric and the X7 audit (≥ 1 Nigeria-based colleague).
- The GPT / Gemini paste-bridge runs for X5.

## Addendum 1 (2026-11-10): fold-in with the locked sensor-actuator programme
Source: `MechInterp/SENSOR_ACTUATOR_PROGRAM_LOCKED_2026-11-10.md`.

**Literature updates that matter here (web, to verify):**
- **No mitigation of geographic diagnostic bias has been evaluated.** A 2026 medRxiv respiratory study across Ghana,
  India, Jordan, Brazil and the UK found that LLMs propose fewer of LMIC doctors' diagnoses (32% vs 50%), and stating
  the country did not fix it. That makes X4 (harness) and X4b the field's first mitigation evaluation, and it
  independently corroborates the NST finding.
- Demographic (not geographic) clinical steering exists: race-HIV activation editing, EMNLP 2026; SAE racial-bias
  steering is weak on realistic tasks. Cite these as the nearest work.

**New and changed experiments:**
- **X4b (Idea A): dual-controlled activation governor vs the X4 harness vs fixed-α steering.**
  - Same split: sensor fitted on 60% of syndromes; X6's new syndromes as the confirmatory set.
  - Reported as a frontier (hard-drop vs capability).
  - Includes a confidence-direction control, given the finding that debiasing directions can just lower confidence.
- **X8 (Idea B sensor): "geographically encoded but not emitted".** When a treatable target is dropped from the
  low-income differential, is it still linearly decodable from the hidden state at the answer position?
  - If yes, the under-call is a routing / suppression failure, not missing knowledge. That sharpens §5 Mechanism and
    justifies activation-level fixes.
  - Includes magnitude and confidence partial-correlation kills. Not MMBU-embargoed.
- **X9 (Idea C, optional, later): constrained medical fine-tuning.** A remedy for §4.1 ("medical fine-tuning can worsen
  it"). Likely a separate paper; cite as ongoing if not ready.

**Revised paper outline (research-paper structure):**
1. Introduction: geographic diagnostic under-call; the resource-adaptation objection stated at full strength.
2. Related work: geographic and demographic bias; resource-stratified guidelines; metamorphic testing; steering and
   harness mitigations.
3. Methods: metamorphic design and taxonomy; benchmark v2 (26 + ~24 held-out syndromes); geography × resource
   factorial (X1); management rubric (X2); prevalence control (X3); mitigations (X4, X4b); mechanism probe (X8);
   matcher and clinician audit (X7).
4. Results:
   - 4.1 under-call across scale, medical tuning and frontier (existing, plus X5 variance);
   - 4.2 the self-judging confound (existing);
   - 4.3 geography, not resources, drives it (X1) + prevalence (X3);
   - 4.4 management adaptation and harmful improvisation (X2);
   - 4.5 mechanism: encoded but not emitted (X8);
   - 4.6 mitigations: harness and governor frontiers (X4, X4b).
5. Discussion: "models adapt the wrong layer"; deployment guidance.
6. Limitations.
7. Conclusion.

**Execution order (unchanged in spirit):**
1. X4 + X4b + X8, all open-model and Kaggle-eligible.
2. X1.
3. X3 (GBD export).
4. X5 (Claude CLI).
5. X2 (rubric).
6. X6 and X7 (clinicians).
- **2026-11-10 X4 RESULT:** blind2pass PASSES the prereg (Qwen2.5-7B, Med42-8B; hard-drop → 0.00, under-call roughly
  halved, over-call not worse). The category-prompted "everywhere" arm fails everywhere. Details in
  NST_X4_X4b_X8_PREREG_2026-11-10.md. This is the field's first evaluated mitigation of geographic diagnostic
  under-call.
