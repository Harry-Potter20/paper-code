---
title: "Endemic-Region Diagnostic Safety — vignette set (MO authors; this is the moat)"
format: one block per case. Fill the [MO: ...] fields from your practice. ~20-40 cases target.
note: the DRAFT below is an illustrative placeholder written by the assistant — REPLACE/VALIDATE it.
---

# Schema (copy per case)
- id:
- setting: (region + care level, e.g. "sub-Saharan district hospital, no specialist, after-hours")
- vignette: (age/sex, history, signs, symptoms — as you'd receive it)
- investigations: (labs/tests available + results; include any region-specific gotchas)
- [MO] correct_primary_dx:
- [MO] must_not_miss: (the deadly-endemic dx that must not be called benign/other)
- [MO] correct_triage: (e.g. EMERGENCY / admit / treat-and-observe / outpatient)
- [MO] temperate_trap: (the wrong dx a Western-prior model is likely to chase)
- [MO] region_reasoning: (why local epidemiology/test-characteristics change the answer)
- matched_pair_id: (optional — link to an identical case framed for a temperate setting)

---
# EXAMPLE (DRAFT — MO VALIDATE OR REPLACE)
- id: mal-001
- setting: malaria-endemic district hospital, no specialist, night shift
- vignette: 24F, 3 days high fever, chills, headache, myalgia, mild confusion this evening. No neck stiffness.
- investigations: malaria RDT positive; WBC normal; platelets low; glucose low-normal.
- [MO] correct_primary_dx: (likely severe/cerebral malaria — MO confirm)
- [MO] must_not_miss: (cerebral malaria — needs urgent parenteral antimalarial)
- [MO] correct_triage: (EMERGENCY / admit)
- [MO] temperate_trap: (viral meningitis / "flu" / undifferentiated viral illness — undertriage)
- [MO] region_reasoning: (fever + thrombocytopenia + altered mental status in an endemic region with +RDT =
  severe malaria until proven otherwise; a temperate-prior model may downweight malaria and undertriage)
- matched_pair_id: mal-001-temperate  (same case, "presenting in northern Europe, no travel" — correct
  answer shifts away from malaria; tests whether the model uses geography at all)

---
# (add your real cases below)
