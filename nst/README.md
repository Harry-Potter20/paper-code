# Off the list: language models drop or bury treatable diagnoses in low-resource settings

Code, pre-registrations and result files for this paper (Eke and Umoize; in submission).

- `preregs/`: pre-registrations with their logged results and job identifiers. These are R3 (16 open models, 3 prompt
  templates), X4/X4b/X8 (mitigation harness, governor, the "encoded but not emitted" readout), the X1/X6 expansion
  plan and the vignette generator spec.
- `data/`: the syndrome and disease definitions and the clinical vignettes used as prompts.
- `jobs/`: the scripts as they were run. `r3_analyze.py` and `nst_displacement_sensitivity.py` produce the headline
  numbers.
- `results/`: raw model outputs (`r3_raw/`, `x5/`), X1/X6 provisional results (`x6/`), and the displacement-adjusted
  sensitivity analysis.

The X1/X6 expansion vignettes are drafts that have not been validated by clinicians, and they are marked provisional
wherever they are used. Open models are used from their own Hugging Face repositories and are not redistributed.
