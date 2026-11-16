# DIRAC-S Experimental Specification v1.0

**Project:** DIRAC — Quantum-Inspired Foundation Model for Medical Imaging  
**Phase:** DIRAC-S scientific validation  
**Modality:** 3D CT imaging  
**Version:** 1.0  
**Date:** 2026-08-11  
**Status:** Pre-registration-style experimental specification

---

## 1. Purpose

DIRAC-S is the compact research-scale implementation of the DIRAC Foundation Model.

Its purpose is **not** to maximize performance on a single clinical task. Its purpose is to test the central scientific hypothesis:

> A structured, quantum-inspired, probabilistic, geometric and topological latent-state representation can provide a more useful and transferable representation of medical images than conventional parameter-matched CNN and ViT architectures.

DIRAC-S is therefore the **scientific proving ground** for the broader DIRAC-S/M/L program.

The experiment must be designed so that positive, null, and negative results are all scientifically interpretable.

---

## 2. Central Research Question

Given a 3D CT volume `X`, can DIRAC learn a general-purpose structured latent state

`X → ψ → ρ → G → T → z`

that captures anatomy, morphology, spatial relationships, topology, uncertainty, and potentially latent dynamics, and transfers effectively to multiple downstream medical-imaging tasks?

---

## 3. Primary Hypotheses

### H1 — Representation
DIRAC-S will provide equal or superior downstream transfer performance compared with parameter-matched CNN and ViT baselines.

### H2 — Data efficiency
DIRAC-S will retain a larger proportion of full-data performance as downstream labeled-data availability decreases.

### H3 — Structural representation
Explicit relational and topological representations will improve performance on tasks requiring anatomical structure and spatial organization.

### H4 — Uncertainty
DIRAC-S uncertainty estimates will demonstrate measurable calibration and correlation with prediction error.

### H5 — Robustness
DIRAC-S representations will be more stable under clinically plausible imaging perturbations and distribution shifts.

### H0 — Null hypothesis
DIRAC-specific inductive biases provide no statistically meaningful advantage over parameter-matched conventional architectures.

The null hypothesis remains a legitimate possible outcome.

---

## 4. Scope

### Included in DIRAC-S v1.0

- 3D CT
- anatomical representation learning
- complex-valued latent representation
- probabilistic/density-state representation
- relational/graph representation
- topology representation
- uncertainty estimation
- reconstruction/masked prediction
- downstream transfer
- controlled robustness testing

### Explicitly deferred

- reinforcement learning
- clinical intervention optimization
- online patient interaction
- multimodal fusion
- MRI/PET/ultrasound
- large-scale longitudinal world modelling
- autonomous diagnosis
- clinical deployment

RL/control remains a future DIRAC layer and is not part of DIRAC-S foundation-model validation.

---

## 5. Conceptual Model

DIRAC-S treats a CT scan as an observation of an underlying anatomical state.

`X → ψ(X)`

where `ψ` is a complex latent field.

The model constructs a quantum-inspired probabilistic state `ρ(X)` with:

`ρ ⪰ 0`

and, where normalized:

`Tr(ρ) = 1`.

The latent state is enriched with relational geometry:

`G = (V,E)`

and topological information:

`T(X)`.

The final structured representation is:

`z = f(ρ,G,T)`.

The foundation model therefore aims to represent:

**anatomy + morphology + geometry + topology + uncertainty**

rather than merely a conventional feature vector.

---

## 6. Scientific Position

DIRAC is **quantum-inspired**, not a claim that human anatomy or medical imaging is literally governed by quantum entanglement at the macroscopic scale.

The terms "quantum state", "density matrix", "complex state", "entanglement-inspired correlation", and related terminology refer to mathematical structures used as inductive biases.

The study must test whether those structures are useful.

No biological quantum-mechanics claim is required for DIRAC to succeed.

---

## 7. Foundation Model vs Task Model

DIRAC-S is the foundation model.

A downstream task model is:

`DIRAC-S + task-specific adapter/head`

Examples:

`z → segmentation decoder`

`z → classification head`

`z → detection head`

The foundation model and downstream heads must be evaluated separately.

---

## 8. DIRAC-S Capability Targets

DIRAC-S should produce a structured representation supporting:

1. anatomical representation
2. morphology representation
3. spatial/relational understanding
4. topology-aware representation
5. probabilistic state representation
6. uncertainty quantification
7. multi-scale representation
8. cross-patient anatomical similarity
9. reconstruction
10. representation-level retrieval
11. anomaly/novelty detection
12. robustness to imaging perturbations
13. out-of-distribution awareness
14. eventual temporal/dynamic extension

The first study does not need to demonstrate every capability clinically. Each capability must instead have a measurable probe.

---

## 9. Experimental Pipeline

```text
                         3D CT
                           |
                           v
                 +-------------------+
                 | Preprocessing/QC  |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 |   DIRAC Encoder   |
                 +---------+---------+
                           |
                           v
                 Complex latent ψ
                           |
                           v
                 Probabilistic ρ
                           |
              +------------+-------------+
              |                          |
              v                          v
        Relational graph G         Topology T
              |                          |
              +------------+-------------+
                           |
                           v
                  Structured latent z
                           |
              +------------+-------------+
              |            |             |
              v            v             v
          Probing      Transfer      Robustness
              |            |             |
              v            v             v
        representation   task heads   uncertainty
          quality                     + OOD
```

---

## 10. Stage 0 — Dataset and Preprocessing

### Modality
Initial validation is restricted to 3D CT.

### Dataset selection
Candidate public datasets should prioritize:

- large sample counts
- 3D CT availability
- patient-level identifiers
- reliable metadata
- anatomical diversity
- reproducible public access
- downstream annotations where available

The final dataset list must be frozen before final evaluation.

### Data splitting
All splits must be **patient-level**.

No slices, series, or volumes from the same patient may cross train/validation/test boundaries.

Recommended:

- training set
- validation set
- held-out internal test set
- external test set where available

### Preprocessing
The pipeline should be deterministic and documented:

1. DICOM/volume ingestion
2. quality-control checks
3. orientation normalization
4. voxel-spacing normalization
5. resampling to the specified working resolution
6. HU-aware intensity handling
7. body-region handling/cropping where justified
8. normalization
9. standardized tensor formatting

Raw metadata remain separate from the learned representation.

---

## 11. Stage I — Foundation Pretraining

DIRAC-S pretraining should initially use largely task-agnostic objectives.

### Objective A — Reconstruction

`X → ρ → X_hat`

The representation must retain clinically meaningful image structure.

### Objective B — Masked/partial observation prediction

`X_masked → ρ → X_hat_missing`

This tests contextual anatomical understanding.

### Objective C — Structural consistency

For a controlled transformation `X' = g(X)` that preserves relevant structure:

`d(ρ_X, ρ_X') → 0`.

Invariance must not destroy clinically meaningful variation.

### Objective D — Topological consistency

Controlled topology-preserving and topology-altering perturbations should be used.

The model should remain stable under topology-preserving transformations and respond to meaningful topology-changing transformations.

### Dynamics

Longitudinal dynamics are **not required for the initial DIRAC-S pretraining run**. Where longitudinal CT data exist, an auxiliary objective may test:

`p(ρ_(t+Δt) | ρ_t)`.

This remains secondary.

---

## 12. Core DIRAC Loss

Initial formulation:

`L_DIRAC = λr L_recon + λm L_masked + λs L_struct + λt L_topology`

Exact coefficients must be fixed before final test evaluation.

Probabilistic-state constraints should be enforced by construction wherever possible:

`ρ ⪰ 0`

`Tr(ρ) = 1`.

The complex representation must be explicitly defined and computationally justified.

---

## 13. Stage II — Representation Evaluation

Before downstream fine-tuning, DIRAC-S is evaluated as a frozen representation.

### Linear probe

Freeze all DIRAC parameters and train a minimal linear predictor.

Purpose: measure directly accessible task-relevant information.

### Nonlinear probe

Train a small MLP.

Purpose: determine whether useful information exists but is not linearly separable.

### Retrieval

Perform nearest-neighbour retrieval in latent space and evaluate anatomical/morphological/pathological similarity where labels permit.

### Clustering

Evaluate whether meaningful anatomical categories or states emerge.

Clustering is not proof of semantic understanding without external validation.

---

## 14. Stage III — Uncertainty Evaluation

DIRAC uncertainty must be evaluated quantitatively using appropriate measures such as:

- predictive entropy
- state entropy
- calibration error
- Brier score
- negative log-likelihood where appropriate
- prediction interval coverage
- uncertainty-error correlation
- selective prediction / risk-coverage curves

Central test:

`uncertainty ↑ as prediction error ↑`

High uncertainty alone is not evidence of useful uncertainty.

---

## 15. Stage IV — Downstream Transfer

At minimum, use three task families.

### Task A — Segmentation
Tests spatial representation, anatomy, boundaries, and multi-scale structure.

Possible metrics:

- Dice
- IoU
- surface distance
- Hausdorff distance where appropriate

### Task B — Classification
Tests global semantic representation and morphology/pathology-related information.

Possible metrics:

- AUROC
- AUPRC
- sensitivity
- specificity
- F1
- calibration

### Task C — Detection / anomaly detection
Tests localized abnormality representation and novelty detection.

Metrics depend on task formulation.

---

## 16. Fine-Tuning Regimes

Every downstream task should be tested under three regimes.

### Regime 1 — Frozen DIRAC

`DIRAC_frozen + small task head`

Primary representation-quality test.

### Regime 2 — Partial fine-tuning

Only specified later DIRAC blocks and the task adapter are trainable.

The exact layer configuration must be documented.

### Regime 3 — Full fine-tuning

All DIRAC parameters and the task adapter are trainable.

This estimates maximum attainable task performance.

---

## 17. Baseline Models

Primary baselines:

1. parameter-matched CNN
2. parameter-matched ViT

They must receive:

- the same modality
- the same preprocessing
- the same splits
- comparable augmentation
- comparable optimization budget
- comparable parameter count
- equivalent downstream task capacity

The comparison is invalid if DIRAC receives substantially more parameters or compute without explicit reporting.

---

## 18. Exact Parameter Matching

For DIRAC-S:

`N_DIRAC-S`

must be calculated exactly from the implementation.

Construct:

`N_CNN ≈ N_DIRAC-S`

and:

`N_ViT ≈ N_DIRAC-S`.

Exact parameter counts must be reported, not inferred from nominal model names.

Accounting must include:

- convolution weights
- normalization parameters
- MLP weights
- attention weights
- biases
- complex projections
- graph layers
- topology embeddings
- dynamics modules
- decoders/adapters where relevant

---

## 19. Ablation Study

Reference:

`D_full`

Minimum ablations:

- `D_-complex`
- `D_-prob`
- `D_-graph`
- `D_-topology`
- `D_-dynamics` where applicable
- `D_classical`

The objective is to identify which components produce measurable benefit.

---

## 20. Robustness

Test controlled perturbations where clinically appropriate:

- noise
- intensity perturbation
- resolution changes
- spatial transformations
- partial information loss
- acquisition/protocol variation
- simulated artifacts

For a clinically irrelevant transformation `g`, ideally:

`d_Z(X,g(X))` remains small.

For meaningful structural change:

`d_Z(X,X_path)` should increase appropriately.

---

## 21. Distribution Shift

Where external datasets exist:

`Train: D_A`

`Test: D_B`

with:

`D_A ≠ D_B`.

Compare degradation:

`Δ_DIRAC` vs `Δ_CNN` vs `Δ_ViT`.

A strong result would be:

`Δ_DIRAC < Δ_baseline`.

---

## 22. Data-Efficiency Experiment

Downstream labels should be evaluated at predetermined fractions:

`1%, 5%, 10%, 25%, 50%, 100%`.

Question:

> Does DIRAC retain useful performance when labels are scarce?

Area under the learning curve may be reported as a secondary summary.

---

## 23. Reproducibility

Major experiments should use multiple random seeds.

Recommended initial minimum:

`n ≥ 3`

where compute permits more.

Report:

- mean
- standard deviation
- confidence intervals where appropriate
- exact seeds
- software versions
- hardware
- training steps
- batch size
- optimizer
- learning-rate schedule
- checkpoint selection criterion

All preprocessing and split-generation code must be version controlled.

---

## 24. Statistical Analysis

The final test set must remain untouched until final evaluation.

Do not tune architecture, hyperparameters, loss weights, preprocessing, or thresholds against the final test set.

Use:

- paired tests where appropriate
- bootstrap confidence intervals where appropriate
- multiple-comparison correction where necessary
- effect sizes in addition to p-values

Statistical significance alone is insufficient.

---

## 25. Success Criteria

### Minimum success
At least one strong, reproducible advantage over parameter-matched baselines.

### Strong success
Advantages across multiple dimensions, such as:

- representation probing
- downstream transfer
- data efficiency
- robustness
- uncertainty/calibration

with supportive ablations.

### Exceptional result
Consistent advantages across task types and external distribution shifts, with ablations showing that the gains derive from identifiable DIRAC components rather than parameter count.

---

## 26. Failure Criteria

### Failure A
DIRAC-S performs worse than matched baselines.

### Failure B
DIRAC-S matches baselines but provides no structural or uncertainty advantage.

### Failure C
DIRAC-S improves only one task and fails broadly.

### Failure D
DIRAC-S appears strong only after extensive full fine-tuning.

These are valid scientific outcomes and must be reported honestly.

---

## 27. Paper Decision Gate

The first paper should be based on the **results of DIRAC-S**, not merely on the existence of the architecture.

A publication-ready study requires:

1. reproducible implementation
2. exact parameter accounting
3. parameter-matched baselines
4. controlled downstream transfer
5. meaningful ablations
6. uncertainty evaluation
7. robustness evaluation
8. statistical analysis
9. a clear result supporting or refuting the central hypothesis

If these conditions are met, DIRAC-S can form the basis of the first paper.

---

## 28. Proposed First Paper

### Working title

**DIRAC: A Quantum-Inspired Probabilistic, Geometric and Topological Foundation Model for Medical Imaging**

### Central claim to test

> DIRAC introduces a structured latent-state formulation for medical imaging that integrates complex-valued representations, probabilistic state modelling, relational geometry, topology and uncertainty estimation, and tests whether these inductive biases improve transferable medical-image representations under controlled parameter-matched evaluation.

The paper must avoid claims of literal quantum biological computation.

---

## 29. Proposed Paper Structure

### Abstract
Problem → method → experimental design → principal result → implication.

### 1. Introduction
- limitations of conventional medical-imaging representation learning
- motivation for structured latent states
- uncertainty and structural representation
- DIRAC hypothesis
- contributions

### 2. Related Work
- medical imaging foundation models
- self-supervised learning
- vision foundation models
- geometric deep learning
- graph neural networks
- topological deep learning
- uncertainty-aware learning
- quantum-inspired machine learning

### 3. DIRAC
- mathematical formulation
- complex representation
- probabilistic state
- relational graph
- topology
- uncertainty
- architecture

### 4. Experimental Protocol
- datasets
- preprocessing
- training
- baselines
- parameter matching
- downstream tasks
- ablations
- statistical protocol

### 5. Results
- representation quality
- transfer
- data efficiency
- uncertainty
- robustness
- OOD
- ablations

### 6. Discussion
- what DIRAC learned
- which components matter
- where it fails
- computational cost
- limitations

### 7. Conclusion
What the DIRAC-S evidence supports and what remains unresolved.

---

## 30. DIRAC-S → DIRAC-M → DIRAC-L

Scaling must be evidence-driven:

`DIRAC-S → validation → DIRAC-M → scaling study → DIRAC-L`

### DIRAC-S
Validate the scientific premise.

### DIRAC-M
Establish a serious medical-imaging foundation model at larger scale.

### DIRAC-L
Test whether the formulation scales with data, model capacity, and heterogeneous imaging distributions.

Scaling is not assumed to improve performance.

---

## 31. Future Work — Explicitly Deferred

- model-based RL
- clinical control
- treatment optimization
- causal intervention modelling
- large-scale longitudinal forecasting
- multimodal medical foundation learning
- autonomous clinical decision-making
- prospective clinical deployment

The eventual conceptual extension is:

`p(ρ_(t+1) | ρ_t, a_t)`

followed by model-based planning/control.

This is a future layer, not a requirement for DIRAC-S.

---

## 32. Core Scientific Philosophy

DIRAC-S should answer one question cleanly:

> **Does explicitly structuring the latent representation of medical images around probabilistic state, complex representation, relational geometry and topology provide measurable advantages over conventional architectures at matched capacity?**

Everything in the first experiment should serve this question.

If yes, scale.

If no, diagnose why.

If only certain components work, simplify DIRAC.

The goal is not to prove the original idea right.

The goal is to discover whether it is right.

---

## 33. Final DIRAC-S Definition

`DIRAC-S = compact quantum-inspired probabilistic foundation model`

for:

`3D medical imaging`

with a structured latent state containing:

`complex representation + probabilistic state + geometry + relational structure + topology + uncertainty`

and designed for:

`transferable representation learning`

rather than a single task.

---

## 34. Immediate Implementation Order

1. Freeze the DIRAC-S mathematical specification.
2. Freeze the exact DIRAC-S architecture.
3. Calculate the exact parameter count.
4. Define every trainable and frozen parameter.
5. Select and freeze the initial public CT datasets.
6. Implement preprocessing.
7. Implement DIRAC-S pretraining.
8. Implement unit tests for the probabilistic state.
9. Implement topology and graph modules.
10. Train DIRAC-S.
11. Construct parameter-matched CNN and ViT baselines.
12. Run representation probes.
13. Run downstream transfer.
14. Run uncertainty experiments.
15. Run robustness/OOD experiments.
16. Run ablations.
17. Repeat critical experiments across seeds.
18. Freeze the final results.
19. Decide whether the evidence supports the first DIRAC paper.
20. Only then begin DIRAC-M scaling.

---

## Non-negotiable Principle

**Do not add architectural components merely because they sound theoretically attractive.**

Every component must answer:

> What capability does this provide?

and:

> What experiment can falsify the claim that it helps?

That principle governs DIRAC-S v1.0.
