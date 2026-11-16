# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "nibabel", "gudhi"]
# ///
# gudhi is a transitive import (model.dirac_model -> model.topology imports it at module level for
# compute_persistence_summary, even though this job only ever calls TopologyHead, never that function) --
# missing it isn't caught by a local dry run if gudhi already happens to be installed in the dev environment.
"""Phase 5 slice 1 (IMPLEMENTATION_SPEC.md): H1 linear probe. "Freeze all DIRAC parameters and train a
minimal linear predictor... measure directly accessible task-relevant information" (experimental-spec
Section 13), run identically against the matched CNN3D/ViT3D baselines (Gate 4 checkpoints) so H1 is a real
comparison, not a solo number.

Label: tumor volume in physical mm^3 (not raw voxel count), derived from the segmentation masks
materialized in Phase 1 (out_dir/segmentation.nii.gz, never read during pretraining — data/leak_guard.py's
invariant is untouched since probing is explicitly the eval stage, spec Section 15) via mask value 2. That
tumor-label-value convention isn't invented here — verified against covtoken's own already-working
ingestion of these exact three datasets (covtoken/jobs/materialize_kits_job.py: "KiTS labels: 1=kidney,
2=tumor, 3=cyst"; materialize_msd_pancreas_job.py: "1=pancreas, 2=tumor... parameterized for any MSD task
with a tumor label (pancreas Task07, liver Task03, ...)"). Two label bugs were caught and fixed by a
pre-lock audit before drawing any conclusion from the first real run's near-chance result:
  1. Raw voxel count is NOT physical volume — segmentation.nii.gz is preserved at each scan's ORIGINAL
     native resolution (materialize_kits_job.py/materialize_msd_job.py both shutil.move the raw downloaded
     mask as-is, never resampled to the 1.5mm isotropic spacing volume.npy uses), so voxel count alone
     conflates true tumor size with each scan's native voxel spacing. Fixed via the mask's own header
     zooms (tumor_volume_mm3 below).
  2. The per-dataset median threshold was computed over the POOLED train+val volumes — leaking validation
     tumor volumes into the very threshold used to label the validation set. Fixed: median computed from
     TRAIN ONLY per dataset, applied to both splits (compute_labels_and_volumes below).
A separate, real (not a bug, not fully fixable here) limitation confirmed via a direct check of Phase 1's
own QC data: data/preprocess.py's resize_to_fixed rescales each case's foreground crop to a fixed 128^3
grid with a PER-CASE zoom factor (crop size varies up to 27x within KiTS23 alone, CV 0.37-0.53 across all
three datasets — confirmed against qc.cropped_shape in the frozen manifests). That means the effective
mm/voxel in what the model actually sees varies per case, proportional to that patient's own body/scan
extent — partially confounding absolute physical tumor size with body/FOV size in the model's input, for
all three architectures equally. This doesn't invalidate the label (mm^3 from the native, unprocessed mask
is still a valid ground truth) but it does lower the ceiling on what any architecture could recover from
its actual input, and is reported alongside the results rather than left implicit.
An even earlier version of this job used tumor presence/absence directly; a real run showed that label is
~98-100% positive on these three datasets (curated tumor-segmentation challenges, not population samples)
— degenerate for probing, replaced with the size-based label before the audit above.

Regime 1 only (frozen backbone + linear head, experimental-spec Section 16) — the primary representation-
quality test. Test split is untouched (experimental-spec Section 24): probes train on train, are scored on
val; the final confirmatory test-split number is reserved for when all of Phase 5 (H1-H5 + ablations) is
locked, not spent on this first slice.

Two probes per architecture, both on the same frozen z: (1) the classification probe (size above/below the
train-only median — the headline H1 comparison), and (2) a continuous regression probe (Linear(256,1) on
log1p(volume_mm3), evaluated via Spearman rank correlation — the primary statistic, since tumor volumes are
strongly right-skewed — plus Pearson r and R^2). The classification split may hide a real monotonic signal
that a binary threshold discards (a 1% and a 10x difference can land on the same side of a median); the
regression diagnostic checks for that directly rather than trusting the binary result alone.

Caches extracted (frozen, deterministic) z features to the bucket at processed/dirac/probe_features/
{arch}_{split}_z.npz — z/case_id only, no label, so probe_nonlinear_job.py (run after this) can reuse the
expensive part (materializing + a forward pass over every case for all three models) regardless of any
future change to the label/threshold logic. An earlier version bundled z+label into one cache file and hit
exactly that problem: fixing the label required either wiping stale caches or risking a silent stale-label
reuse. Labels/volumes are cheap to recompute from the masks every run, so they're never cached.

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

from model.dirac_model import DiracModel
from baselines.cnn3d import CNN3DBaseline
from baselines.vit3d import ViT3DBaseline
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import auroc, delong_auc_ci, delong_auc_diff_test, spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_probe"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256          # DIRAC-S, locked
CNN_WIDTH, CNN_DEPTH = 36, 4                               # CNN3DBaseline, locked
VIT_WIDTH, VIT_DEPTH, VIT_HEADS, VIT_PATCH, VOLUME_SIZE = 136, 4, 4, 16, 128   # ViT3DBaseline, locked
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
TUMOR_LABEL_VALUE = 2

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-linear] {m}", flush=True)


def load_rows() -> list[dict]:
    """All qc_passed train+val cases — test stays untouched (spec Section 24)."""
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def tumor_volume_mm3(row: dict) -> float | None:
    """Physical tumor volume in mm^3 -- NOT raw voxel count. segmentation.nii.gz is preserved at each
    scan's original native resolution (never resampled to volume.npy's shared 1.5mm isotropic spacing),
    so raw voxel count would conflate true tumor size with each scan's own voxel spacing. The mask's own
    header zooms (voxel dimensions in mm) convert voxel count to a properly comparable physical volume."""
    seg_path = Path(row["local_seg_path"])
    if not seg_path.exists():
        return None
    img = nib.load(str(seg_path))
    mask = img.get_fdata()
    voxel_count = int((mask == TUMOR_LABEL_VALUE).sum())
    dx, dy, dz = img.header.get_zooms()[:3]
    return voxel_count * float(dx) * float(dy) * float(dz)


def compute_labels_and_volumes(rows: list[dict]) -> tuple[dict[str, int], dict[str, float]]:
    """Binary label = physical tumor volume above/below the median, split per-dataset (not pooled
    globally, to avoid confounding size with organ identity) AND computed from TRAIN cases only, then
    applied to both train and val -- computing the threshold on the pooled train+val set would leak
    validation tumor volumes into the very threshold used to score validation, which is exactly what
    happened before this audit. Returns (binary_labels, continuous_volumes_mm3); the continuous value is
    kept so a rank-correlation/regression diagnostic can check whether the binary split is hiding a real
    monotonic signal (formalization/spec discipline: don't trust a threshold's null result without also
    checking the un-thresholded relationship)."""
    volumes: dict[str, float] = {}
    dataset_of: dict[str, str] = {}
    split_of: dict[str, str] = {}
    for row in rows:
        v = tumor_volume_mm3(row)
        if v is None:
            continue
        volumes[row["case_id"]] = v
        dataset_of[row["case_id"]] = row["dataset"]
        split_of[row["case_id"]] = row["split"]

    train_by_dataset: dict[str, list[float]] = {}
    for case_id, v in volumes.items():
        if split_of[case_id] == "train":
            train_by_dataset.setdefault(dataset_of[case_id], []).append(v)
    median_by_dataset = {ds: sorted(vs)[len(vs) // 2] for ds, vs in train_by_dataset.items()}

    labels = {cid: int(v > median_by_dataset[dataset_of[cid]]) for cid, v in volumes.items()}
    return labels, volumes


def load_dirac() -> DiracModel:
    ckpt = torch.load(MNT / "processed" / "dirac" / "checkpoints" / "pretrain_full_seed0.pt", map_location="cpu")
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=TOPOLOGY_MAX_DIM, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_cnn() -> CNN3DBaseline:
    ckpt = torch.load(MNT / "processed" / "dirac" / "checkpoints" / "baseline_cnn_full_seed0.pt", map_location="cpu")
    model = CNN3DBaseline(width=CNN_WIDTH, depth=CNN_DEPTH, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_vit() -> ViT3DBaseline:
    ckpt = torch.load(MNT / "processed" / "dirac" / "checkpoints" / "baseline_vit_full_seed0.pt", map_location="cpu")
    model = ViT3DBaseline(width=VIT_WIDTH, depth=VIT_DEPTH, heads=VIT_HEADS, patch=VIT_PATCH,
                          volume_size=VOLUME_SIZE, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def feature_cache_path(arch: str, split: str) -> Path:
    # "_z" (no label bundled in) -- label/threshold logic can change freely without ever risking a silent
    # stale-label reuse; see the module docstring for why this is a deliberate fix, not just a rename.
    return MNT / "processed" / "dirac" / "probe_features" / f"{arch}_{split}_z.npz"


def extract_features(arch: str, model, rows: list[dict], device, ph_vectors: dict[str, np.ndarray] | None) -> None:
    """Runs one forward pass per case (no_grad), writes cached (z, case_id) arrays per split. Extracts for
    every row regardless of label availability -- z depends only on the frozen model and the input volume,
    never on the label, so there's no reason to gate extraction on whether a usable mask exists."""
    by_split: dict[str, list] = {"train": [], "val": []}
    model = model.to(device)
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,128,128,128)
            if arch == "dirac":
                ph = torch.from_numpy(ph_vectors[row["local_path"]]).unsqueeze(0).to(device)
                z = model(x, ph_vector=ph)["z"]
            else:
                z = model(x)
            by_split[row["split"]].append((row["case_id"], z.squeeze(0).cpu().numpy()))
            if i % 50 == 0:
                log(f"{arch}: extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        if not items:
            continue
        case_ids = [it[0] for it in items]
        z = np.stack([it[1] for it in items])
        out_path = feature_cache_path(arch, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, z=z, case_id=np.array(case_ids))
        log(f"{arch}/{split}: cached {len(items)} cases -> {out_path}")


def get_features(arch: str, model_loader, rows: list[dict], device, ph_vectors) -> dict[str, dict]:
    if feature_cache_path(arch, "train").exists() and feature_cache_path(arch, "val").exists():
        log(f"{arch}: using cached features")
    else:
        extract_features(arch, model_loader(), rows, device, ph_vectors)
    out = {}
    for split in ("train", "val"):
        d = np.load(feature_cache_path(arch, split))
        out[split] = {"z": d["z"], "case_id": d["case_id"]}
    return out


def attach_labels(features: dict[str, dict], case_labels: dict[str, int],
                  case_volumes: dict[str, float]) -> dict[str, dict]:
    """Filters each split's (z, case_id) down to cases with a usable label, attaching both the binary
    label and the continuous log1p(volume_mm3) regression target -- log1p (not raw mm^3) since tumor
    volumes are strongly right-skewed and log1p is well-defined at volume=0 (the small number of
    zero-tumor cases the median split still includes)."""
    out = {}
    for split, d in features.items():
        keep = [i for i, cid in enumerate(d["case_id"]) if cid in case_labels]
        case_ids = d["case_id"][keep]
        out[split] = {
            "z": d["z"][keep],
            "case_id": case_ids,
            "label": np.array([case_labels[cid] for cid in case_ids], dtype=np.int64),
            "log_volume": np.array([np.log1p(case_volumes[cid]) for cid in case_ids], dtype=np.float32),
        }
    return out


class LinearProbe(nn.Module):
    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.linear = nn.Linear(z_dim, 1)

    def forward(self, z):
        return self.linear(z).squeeze(-1)


def train_probe(probe: nn.Module, z_train: np.ndarray, label_train: np.ndarray, device) -> nn.Module:
    probe = probe.to(device)
    z = torch.from_numpy(z_train).float().to(device)
    y = torch.from_numpy(label_train).float().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(probe(z), y)
        loss.backward()
        opt.step()
    return probe.eval()


def score(probe: nn.Module, z: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        logits = probe(torch.from_numpy(z).float().to(device))
        return torch.sigmoid(logits).cpu().numpy()


def probe_head_factory():
    """Overridden by probe_nonlinear_job.py's near-identical copy — kept as a single seam so the two
    files' only structural difference is this one function, not the surrounding extraction/eval logic."""
    return LinearProbe()


def train_regression_probe(probe: nn.Module, z_train: np.ndarray, y_train: np.ndarray, device) -> nn.Module:
    """Same probe architecture as the classifier, trained with MSE against log1p(volume_mm3) instead of
    BCE against the binary label -- checks whether the binary median split is hiding a real monotonic
    signal (formalization/spec's own H0-honesty discipline: a null classification result shouldn't be
    trusted without also checking the un-thresholded relationship, since a threshold can throw away a
    1%-vs-10x distinction that lands on the same side of the median)."""
    probe = probe.to(device)
    z = torch.from_numpy(z_train).float().to(device)
    y = torch.from_numpy(y_train).float().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(probe(z), y)
        loss.backward()
        opt.step()
    return probe.eval()


def predict_regression(probe: nn.Module, z: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        return probe(torch.from_numpy(z).float().to(device)).cpu().numpy()


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R^2 (sklearn's definition: 1 - SS_res/SS_tot, SS_tot from y_true's own mean) -- can be
    negative if the probe does worse than predicting the mean, which is a real, reportable outcome here,
    not a bug."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    seg_rows = [{**r, "out_path": str(Path(r["out_path"]).parent / "segmentation.nii.gz")} for r in rows]
    seg_rows = materialize_local(seg_rows, bucket_root=MNT, local_root=LOCAL)
    for r, sr in zip(rows, seg_rows):
        r["local_seg_path"] = sr["local_path"]
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_vectors = {r["local_path"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    case_labels, case_volumes = compute_labels_and_volumes(rows)
    log(f"{len(case_labels)} cases with a usable label "
       f"({sum(case_labels.values())} above-median 'large', {len(case_labels) - sum(case_labels.values())} 'small', "
       f"median computed on train only per dataset)")

    t0 = time.time()
    raw_features = {
        "dirac": get_features("dirac", load_dirac, rows, device, ph_vectors),
        "cnn": get_features("cnn", load_cnn, rows, device, ph_vectors),
        "vit": get_features("vit", load_vit, rows, device, ph_vectors),
    }
    log(f"feature extraction/loading done in {time.time() - t0:.1f}s")
    features = {arch: attach_labels(f, case_labels, case_volumes) for arch, f in raw_features.items()}

    results = {}
    val_scores = {}
    val_labels = None
    for arch in ("dirac", "cnn", "vit"):
        train, val = features[arch]["train"], features[arch]["val"]

        probe = train_probe(probe_head_factory(), train["z"], train["label"], device)
        s = score(probe, val["z"], device)
        auc, ci = delong_auc_ci(s, val["label"])

        reg = train_regression_probe(probe_head_factory(), train["z"], train["log_volume"], device)
        pred = predict_regression(reg, val["z"], device)
        spearman = spearman_perm(pred, val["log_volume"], n=2000, seed=SEED)
        pearson = pearson_r(pred, val["log_volume"])
        r2 = r_squared(val["log_volume"], pred)

        results[arch] = {
            "n_train": len(train["label"]), "n_val": len(val["label"]),
            "n_train_above_median": int(train["label"].sum()), "n_val_above_median": int(val["label"].sum()),
            "classification": {"val_auroc": auc, "val_auroc_ci95": list(ci)},
            "regression_log_volume": {"spearman_rho": spearman["rho"], "spearman_p": spearman["p"],
                                      "pearson_r": pearson, "r_squared": r2},
        }
        val_scores[arch] = s
        val_labels = val["label"]
        log(f"{arch}: val AUROC={auc:.4f} CI95={ci} | log-volume spearman={spearman['rho']:.4f} "
           f"(p={spearman['p']:.4f}) pearson={pearson:.4f} R2={r2:.4f}")

    comparisons = {}
    for baseline in ("cnn", "vit"):
        cmp = delong_auc_diff_test(val_scores["dirac"], val_scores[baseline], val_labels)
        comparisons[f"dirac_vs_{baseline}"] = cmp
        log(f"dirac vs {baseline}: diff={cmp['diff']:.4f} CI95={cmp['ci95']} p={cmp['p']:.4f}")

    report = {
        "hypothesis": "H1", "probe_type": "linear",
        "label": "tumor_volume_mm3 (physical units, from native-resolution mask header zooms), binarized "
                "above/below the TRAIN-ONLY median split per-dataset for classification, and used directly "
                "(log1p-transformed) for the regression diagnostic. mask value 2 = tumor, verified against "
                "covtoken's ingestion of the same 3 datasets. See module docstring for the two label bugs "
                "a pre-lock audit caught and fixed (raw voxel count vs physical mm^3; train+val-pooled vs "
                "train-only median) and the confirmed (not fully fixable here) per-case scale-distortion "
                "finding from data/preprocess.py's fixed-grid resize.",
        "regime": "1 (frozen backbone + linear head)",
        "split_policy": "probe trained on train split, scored on val split; test split untouched (spec Section 24)",
        "results": results,
        "comparisons": comparisons,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / "probe_linear_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_LINEAR_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-linear] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
