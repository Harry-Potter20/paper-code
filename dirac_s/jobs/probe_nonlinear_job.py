# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "nibabel", "gudhi"]
# ///
# gudhi is a transitive import (model.dirac_model -> model.topology imports it at module level) — see
# probe_linear_job.py's header comment for the full explanation.
"""Phase 5 slice 1 (IMPLEMENTATION_SPEC.md): H1 nonlinear probe. "Train a small MLP... determine whether
useful information exists but is not linearly separable" (experimental-spec Section 13) — same label,
splits, regime and statistical comparison as probe_linear_job.py (see that file's docstring for the full
rationale, including the two label bugs a pre-lock audit caught and fixed and the confirmed per-case
scale-distortion finding from data/preprocess.py's fixed-grid resize); the only structural difference is
the probe head itself (probe_head_factory below) plus the same classification-vs-regression pair of probes
per architecture (does a nonlinear head recover more signal than the linear one, on either diagnostic).

Reuses probe_linear_job.py's cached z features at processed/dirac/probe_features/{arch}_{split}_z.npz if
that job already ran (deterministic frozen-model features, safe to share, and intentionally decoupled from
any label/threshold logic so a label fix never risks a silent stale-label reuse) — falls back to extracting
them itself if run standalone, so this file works on its own too.

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
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "64"))


def log(m):
    print(f"[probe-nonlinear] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def tumor_volume_mm3(row: dict) -> float | None:
    """Physical tumor volume in mm^3 -- see probe_linear_job.py's identical function for the full
    rationale (raw voxel count would conflate true size with each scan's own native voxel spacing)."""
    seg_path = Path(row["local_seg_path"])
    if not seg_path.exists():
        return None
    img = nib.load(str(seg_path))
    mask = img.get_fdata()
    voxel_count = int((mask == TUMOR_LABEL_VALUE).sum())
    dx, dy, dz = img.header.get_zooms()[:3]
    return voxel_count * float(dx) * float(dy) * float(dz)


def compute_labels_and_volumes(rows: list[dict]) -> tuple[dict[str, int], dict[str, float]]:
    """Binary label = physical tumor volume above/below the TRAIN-ONLY median, split per-dataset -- see
    probe_linear_job.py's identical function for the full rationale (train+val-pooled median leaks
    validation volumes into the val threshold)."""
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
    return MNT / "processed" / "dirac" / "probe_features" / f"{arch}_{split}_z.npz"


def extract_features(arch: str, model, rows: list[dict], device, ph_vectors: dict[str, np.ndarray] | None) -> None:
    by_split: dict[str, list] = {"train": [], "val": []}
    model = model.to(device)
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
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


class MLPProbe(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


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
    return MLPProbe()


def train_regression_probe(probe: nn.Module, z_train: np.ndarray, y_train: np.ndarray, device) -> nn.Module:
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
        "hypothesis": "H1", "probe_type": "nonlinear",
        "label": "tumor_volume_mm3 (physical units, from native-resolution mask header zooms), binarized "
                "above/below the TRAIN-ONLY median split per-dataset for classification, and used directly "
                "(log1p-transformed) for the regression diagnostic. See probe_linear_job.py's module "
                "docstring for the full audit rationale.",
        "regime": "1 (frozen backbone + MLP head)",
        "split_policy": "probe trained on train split, scored on val split; test split untouched (spec Section 24)",
        "results": results,
        "comparisons": comparisons,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / "probe_nonlinear_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_NONLINEAR_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-nonlinear] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
