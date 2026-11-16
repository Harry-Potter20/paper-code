# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "nibabel"]
# ///
"""Phase 5 slice 3 (IMPLEMENTATION_SPEC.md): H3-A (geometry) and H3-B (relations), the remaining two
sub-hypotheses of H3 (Structural Representation) -- run together since both use the exact same pre-fusion
`h` representations already extracted and cached by jobs/probe_h3_topology_job.py (h3c_{dirac,cnn,vit}_h_c1_
{split}.npz), avoiding a second, redundant GPU extraction pass over the same 678 cases.

Unlike H3-C (topology), neither target here has an injection-confound problem: nothing in GraphHead or the
geometric structure of any of the three architectures is fed a per-case ground-truth side-channel the way
ph_vector is for DIRAC's topology_head. So this is a single, fair h-level comparison across all three
architectures -- no C0/C1 split needed.

Targets, both derived from the segmentation masks already materialized in Phase 1 (never read during
pretraining -- data/leak_guard.py's invariant untouched, probing is the eval stage), and both expressed as
FRACTIONS of the mask's own shape rather than absolute physical units. That choice isn't arbitrary: Gate
5's H1 audit found data/preprocess.py's resize_to_fixed applies a PER-CASE zoom factor, so absolute physical
size/position is confounded with each patient's own crop-region size. A fractional/relative coordinate is
invariant to a uniform per-axis rescale by construction, so it sidesteps that confound rather than
inheriting it -- deliberately chosen for exactly that reason, not just convenience.

  H3-A -- geometry (single-entity spatial properties): tumor centroid position (3 dims, z/y/x as a fraction
  of the mask's own shape) + tumor bounding-box extent (3 dims, bbox size as a fraction of the mask's own
  shape per axis). 6 dims total.

  H3-B -- relations (two-entity spatial relationship): organ-to-tumor displacement vector (3 dims,
  organ_centroid - tumor_centroid, fraction of mask shape) + the displacement's magnitude (1 dim, the
  normalized distance). 4 dims total. Tests whether the representation captures the SPATIAL RELATIONSHIP
  between two distinct anatomical entities (organ, label 1; tumor, label 2), not just either one's own
  location -- a genuinely different, higher-level target than H3-A.

Cases with zero tumor voxels or zero organ voxels are skipped (both targets undefined) -- the same small
presence-negative minority documented in probe_linear_job.py's docstring, no special-casing needed.

Same probe architecture/optimization/evaluation-metric as H3-C for direct comparability: single
Linear(dim_in, n_dims) per architecture per hypothesis, MSE loss, Adam lr=1e-2, 300 epochs, evaluated via
per-dimension Spearman rank correlation (primary) and R^2, plus a macro-average Spearman.

Run with: -v hf://datasets/Chucks90/dirac-code:/dirac_pkg:ro -v hf://buckets/Chucks90/jobs-artifacts:/mnt
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/dirac_pkg")

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from data.local_cache import materialize_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_h3ab"))
DATASETS = ["kits", "pancreas", "liver"]
ORGAN_LABEL_VALUE, TUMOR_LABEL_VALUE = 1, 2
GEOMETRY_DIM_NAMES = ["centroid_z", "centroid_y", "centroid_x", "extent_z", "extent_y", "extent_x"]
RELATION_DIM_NAMES = ["displacement_z", "displacement_y", "displacement_x", "distance"]

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-h3ab] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def geometry_and_relation_targets(row: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    seg_path = Path(row["local_seg_path"])
    if not seg_path.exists():
        return None, None
    mask = nib.load(str(seg_path)).get_fdata()
    shape = np.array(mask.shape, dtype=np.float32)

    tumor_coords = np.argwhere(mask == TUMOR_LABEL_VALUE)
    organ_coords = np.argwhere(mask == ORGAN_LABEL_VALUE)
    if len(tumor_coords) == 0 or len(organ_coords) == 0:
        return None, None

    tumor_centroid = tumor_coords.mean(axis=0)
    organ_centroid = organ_coords.mean(axis=0)
    tumor_extent = (tumor_coords.max(axis=0) - tumor_coords.min(axis=0) + 1)

    geometry = np.concatenate([tumor_centroid / shape, tumor_extent / shape]).astype(np.float32)
    displacement = (organ_centroid - tumor_centroid) / shape
    distance = np.linalg.norm(displacement, keepdims=True)
    relation = np.concatenate([displacement, distance]).astype(np.float32)
    return geometry, relation


def h3c_cache_path(condition: str, split: str) -> Path:
    """Reuses jobs/probe_h3_topology_job.py's cache directly -- same deterministic pre-fusion h,
    already extracted, no reason to pay for a second GPU pass over the same 678 cases."""
    return MNT / "processed" / "dirac" / "probe_features" / f"h3c_{condition}_h_c1_{split}.npz"


def load_h_representations() -> dict[str, dict[str, dict]]:
    out = {}
    for arch in ("dirac", "cnn", "vit"):
        out[arch] = {}
        for split in ("train", "val"):
            path = h3c_cache_path(arch, split)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- run jobs/probe_h3_topology_job.py first, this job reuses its "
                    f"cached pre-fusion h representations rather than re-extracting them.")
            d = np.load(path)
            out[arch][split] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


def attach_targets(rep: dict[str, dict], targets_by_case: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for split, d in rep.items():
        keep = [i for i, cid in enumerate(d["case_id"]) if cid in targets_by_case]
        case_ids = d["case_id"][keep]
        out[split] = {"rep": d["rep"][keep], "case_id": case_ids,
                     "target": np.stack([targets_by_case[cid] for cid in case_ids])}
    return out


class LinearRegressionProbe(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)


def train_probe(dim_in: int, dim_out: int, x_train: np.ndarray, y_train: np.ndarray, device) -> nn.Module:
    probe = LinearRegressionProbe(dim_in, dim_out).to(device)
    x = torch.from_numpy(x_train).float().to(device)
    y = torch.from_numpy(y_train).float().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(probe(x), y)
        loss.backward()
        opt.step()
    return probe.eval()


def predict(probe: nn.Module, x: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        return probe(torch.from_numpy(x).float().to(device)).cpu().numpy()


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NaN (not a wild ratio) when ss_tot is too small to be numerically meaningful -- a target with
    genuinely tiny but technically-nonzero variance (e.g. float32 rounding noise on an otherwise-constant
    quantity) can slip past evaluate()'s own upstream degenerate-variance guard and then blow up here:
    ss_res stays a normal-sized number (dominated by the probe's ordinary prediction error) while ss_tot
    collapses toward zero, giving an astronomically negative "R^2" that looks like a real number but isn't
    one. Caught by a local dry run before any GPU spend, not a hypothetical."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else float("nan")


def evaluate(name: str, dim_names: list[str], data: dict[str, dict], device) -> dict:
    train, val = data["train"], data["val"]
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    probe = train_probe(dim_in, dim_out, train["rep"], train["target"], device)
    pred = predict(probe, val["rep"], device)

    per_dim = {}
    rhos = []
    for j, dim_name in enumerate(dim_names):
        y_true, y_pred = val["target"][:, j], pred[:, j]
        if np.std(y_true) < 1e-4:   # scale-aware: fractional targets live in [0,1], not [0,huge]
            per_dim[dim_name] = {"spearman_rho": None, "spearman_p": None, "r_squared": None, "note": "degenerate target (near-zero variance)"}
            continue
        sp = spearman_perm(y_pred, y_true, n=2000, seed=SEED)
        per_dim[dim_name] = {"spearman_rho": sp["rho"], "spearman_p": sp["p"], "r_squared": r_squared(y_true, y_pred)}
        rhos.append(sp["rho"])

    macro_spearman = float(np.mean(rhos)) if rhos else float("nan")
    log(f"{name}: dim_in={dim_in} macro_spearman={macro_spearman:.4f} (n_dims_evaluated={len(rhos)}/{len(dim_names)})")
    return {"dim_in": dim_in, "n_train": len(train["target"]), "n_val": len(val["target"]),
           "macro_spearman": macro_spearman, "per_dim": per_dim}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")   # cheap linear probes on already-extracted features, no GPU needed
    log(f"device={device}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    seg_rows = [{**r, "out_path": str(Path(r["out_path"]).parent / "segmentation.nii.gz")} for r in rows]
    seg_rows = materialize_local(seg_rows, bucket_root=MNT, local_root=LOCAL)
    for r, sr in zip(rows, seg_rows):
        r["local_seg_path"] = sr["local_path"]

    geometry_by_case, relation_by_case = {}, {}
    n_skipped = 0
    for row in rows:
        geom, rel = geometry_and_relation_targets(row)
        if geom is None:
            n_skipped += 1
            continue
        geometry_by_case[row["case_id"]] = geom
        relation_by_case[row["case_id"]] = rel
    log(f"{len(geometry_by_case)} cases with both organ+tumor present ({n_skipped} skipped: missing organ or tumor mask)")

    t0 = time.time()
    h_rep = load_h_representations()
    log(f"loaded cached h representations in {time.time() - t0:.1f}s")

    geometry_data = {arch: attach_targets(rep, geometry_by_case) for arch, rep in h_rep.items()}
    relation_data = {arch: attach_targets(rep, relation_by_case) for arch, rep in h_rep.items()}

    geometry_results = {arch: evaluate(f"H3-A/{arch}", GEOMETRY_DIM_NAMES, d, device) for arch, d in geometry_data.items()}
    relation_results = {arch: evaluate(f"H3-B/{arch}", RELATION_DIM_NAMES, d, device) for arch, d in relation_data.items()}

    report = {
        "hypotheses": {
            "H3-A": "geometry (single-entity spatial properties: tumor centroid + bounding-box extent, both fractional)",
            "H3-B": "relations (two-entity spatial relationship: organ-to-tumor displacement + distance, fractional)",
        },
        "no_injection_confound": "Unlike H3-C, neither target is fed to any architecture as an input at any point -- a single fair h-level comparison, no C0/C1 split needed.",
        "probe": "single Linear(dim_in, n_dims) per architecture per hypothesis, MSE loss, Adam lr=%.0e, %d epochs, seed=%d" % (PROBE_LR, PROBE_EPOCHS, SEED),
        "split_policy": "probe trained on train split, scored on val split; test split untouched (spec Section 24)",
        "representation_source": "reused jobs/probe_h3_topology_job.py's cached pre-fusion h (h3c_{arch}_h_c1_{split}.npz) -- same deterministic representations, no re-extraction",
        "H3-A_geometry_results": geometry_results,
        "H3-B_relation_results": relation_results,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / f"probe_h3ab_geometry_relations_results_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_H3AB_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-h3ab] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
