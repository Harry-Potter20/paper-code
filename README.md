# Paper code

Code, pre-registrations and result files for papers by Chukwudi Eke.

| folder | paper | status |
|---|---|---|
| `dirac_s/` | *Structure or supervision? A parameter-matched, ablation-disentangled test of a structured-latent 3D CT encoder* | in submission |
| `green_means_go/` | *Green means go: an ECG foundation model over-calls normal athletes, and athlete-criteria labels correct it* | in submission |
| `nst/` | *Off the list: language models drop or bury treatable diagnoses in low-resource settings* | in submission |

Each folder holds the job scripts as they were run, the pre-registrations and gate reports (with job identifiers), and
aggregate result files. No model weights, restricted data, raw signals or images are included; `green_means_go/results/`
holds per-ECG model outputs on the public datasets, keyed by their public record identifiers. Public datasets
(KiTS23, Medical Segmentation Decathlon, PTB-XL, Chapman/Ningbo, the Norwegian Endurance Athlete ECG Database,
PF12RED) must be obtained from their own sources under their licences. ECGFounder is used from its own repository
(MIT licence) and is not redistributed here.

Code in this repository is released under the MIT licence (see `LICENSE`).
