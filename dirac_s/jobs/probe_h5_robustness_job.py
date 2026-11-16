# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 5 slice 6 (IMPLEMENTATION_SPEC.md): H5, robustness. Experimental-spec Section 20: test controlled
perturbations (noise, intensity, resolution, partial information loss) where clinically appropriate; for a
clinically irrelevant transformation g, "d_Z(X,g(X)) should remain small." Gate 5's own row: "degradation
delta Delta_DIRAC vs Delta_baseline under matched perturbation severity."

Full distribution-shift testing (Section 21, train on dataset D_A, test on genuinely different D_B) is NOT
attempted here -- flagged since H1's audit as a real, unaddressed gap: all three materialized datasets
(KiTS23, MSD Pancreas, MSD Liver) are similar oncology-CT tumor-segmentation challenges, so there is no
genuinely different D_B available in this vault's current data to test against. This slice covers Section
20's controlled within-distribution perturbations only, which ARE testable with what's already materialized.

Design: reuses H3-C's exact probe methodology for continuity/comparability (same topology target, same
Linear(dim_in, 9) probe architecture, same train split). A probe is trained ONCE per architecture on CLEAN
train h (H3-C's cache, no re-extraction), then scored on (a) CLEAN val h (H3-C's cache, the baseline) and
(b) PERTURBED val h (a fresh forward pass through the frozen model on perturbed input volumes -- cannot be
cached from any prior job). Degradation delta = clean_macro_spearman - perturbed_macro_spearman, smaller is
more robust. Both DIRAC and the baselines see IDENTICAL perturbed volumes at each severity, for a fair,
matched-severity comparison (experimental-spec Section 17's "same modality/preprocessing/splits" extended to
"same perturbation").

Four perturbation families, two severities each (low/high), calibrated to the normalized [-1,1] intensity
range (data/preprocess.py's clip_and_normalize_hu):
  - gaussian_noise: additive N(0, sigma^2), sigma in {0.05, 0.15}
  - intensity_shift: x*scale+offset (mimics scanner/protocol variation), (scale,offset) in
    {(0.9,0.05), (0.7,0.15)}
  - blur: scipy.ndimage.gaussian_filter (mimics lower effective resolution), sigma_vox in {1.0, 2.5}
  - partial_dropout: reuses model/losses.py's generate_block_mask/upsample_token_mask_to_voxels at a LOW
    ratio (distinct from training's 0.75 MAE ratio, which is a training objective, not a robustness probe),
    mask_ratio in {0.10, 0.25}

No DIRAC-S modification: frozen checkpoints only, N_DIRAC-S stays 1,552,520.

Run with: -v hf://datasets/Chucks90/dirac-code:/dirac_pkg:ro -v hf://buckets/Chucks90/jobs-artifacts:/mnt
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/dirac_pkg")

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from model.dirac_model import DiracModel
from model.losses import generate_block_mask, upsample_token_mask_to_voxels
from baselines.cnn3d import CNN3DBaseline
from baselines.vit3d import ViT3DBaseline
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_h5"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256
CNN_WIDTH, CNN_DEPTH = 36, 4
VIT_WIDTH, VIT_DEPTH, VIT_HEADS, VIT_PATCH, VOLUME_SIZE = 136, 4, 4, 16, 128
STEM_STRIDE = 16
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
PH_DIM_NAMES = [f"H{d}_{s}" for d in range(TOPOLOGY_MAX_DIM + 1) for s in ("count", "sum", "max")]

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))

PERTURBATIONS = {
    "gaussian_noise_low": ("gaussian_noise", 0.05), "gaussian_noise_high": ("gaussian_noise", 0.15),
    "intensity_shift_low": ("intensity_shift", (0.9, 0.05)), "intensity_shift_high": ("intensity_shift", (0.7, 0.15)),
    "blur_low": ("blur", 1.0), "blur_high": ("blur", 2.5),
    "partial_dropout_low": ("partial_dropout", 0.10), "partial_dropout_high": ("partial_dropout", 0.25),
}


def log(m):
    print(f"[probe-h5] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def h3c_cache_path(condition: str, split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"h3c_{condition}_h_c1_{split}.npz"


def load_train_h() -> dict[str, dict]:
    out = {}
    for arch in ("dirac", "cnn", "vit"):
        path = h3c_cache_path(arch, "train")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run jobs/probe_h3_topology_job.py first.")
        d = np.load(path)
        out[arch] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


def load_clean_val_h() -> dict[str, dict]:
    out = {}
    for arch in ("dirac", "cnn", "vit"):
        d = np.load(h3c_cache_path(arch, "val"))
        out[arch] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


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


def apply_perturbation(x: torch.Tensor, kind: str, param, device, seed: int) -> torch.Tensor:
    """x: (B,1,D,H,W), normalized to [-1,1]. Deterministic given `seed` (matched across architectures --
    every architecture sees the identical perturbed volume at a given case/severity)."""
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if kind == "gaussian_noise":
        noise = torch.randn(x.shape, generator=gen, device=x.device) * param
        return (x + noise).clamp(-1.0, 1.0)
    if kind == "intensity_shift":
        scale, offset = param
        return (x * scale + offset).clamp(-1.0, 1.0)
    if kind == "blur":
        vol = x.squeeze(0).squeeze(0).cpu().numpy()
        blurred = ndimage.gaussian_filter(vol, sigma=param)
        return torch.from_numpy(blurred).unsqueeze(0).unsqueeze(0).to(device)
    if kind == "partial_dropout":
        grid = (x.shape[2] // STEM_STRIDE,) * 3
        token_mask = generate_block_mask(grid, mask_ratio=param, batch=1, device=device, generator=gen)
        voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
        return x.masked_fill(voxel_mask, 0.0)
    raise ValueError(kind)


def extract_h_perturbed(arch: str, model, rows: list[dict], device, perturb_kind: str, perturb_param) -> dict:
    model = model.to(device)
    case_ids, reps = [], []
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            x = apply_perturbation(x, perturb_kind, perturb_param, device, seed=hash(row["case_id"]) % (2**31))
            if arch == "dirac":
                h, _ = model.encoder.encode_hidden(x)
                pooled = h.mean(dim=1)
            else:
                out = model.encode_hidden(x)
                feat = out[0] if isinstance(out, tuple) else out
                pool_dims = (1,) if feat.dim() == 3 else (2, 3, 4)
                pooled = feat.mean(dim=pool_dims)
            case_ids.append(row["case_id"])
            reps.append(pooled.squeeze(0).cpu().numpy())
    return {"rep": np.stack(reps), "case_id": np.array(case_ids)}


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


def score(probe: nn.Module, rep: np.ndarray, target: np.ndarray, device, seed: int) -> dict:
    with torch.no_grad():
        pred = probe(torch.from_numpy(rep).float().to(device)).cpu().numpy()
    rhos = []
    for j, dim_name in enumerate(PH_DIM_NAMES):
        y_true, y_pred = target[:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=1000, seed=seed)
        rhos.append(sp["rho"])
    return {"macro_spearman": float(np.mean(rhos)) if rhos else float("nan")}


def align(rep: dict, target_by_case: dict[str, np.ndarray]) -> dict:
    keep = [i for i, cid in enumerate(rep["case_id"]) if cid in target_by_case]
    case_ids = rep["case_id"][keep]
    return {"rep": rep["rep"][keep], "case_id": case_ids,
           "target": np.stack([target_by_case[cid] for cid in case_ids])}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    rows = load_rows()
    val_rows = [r for r in rows if r["split"] == "val"]
    log(f"{len(rows)} qc_passed train+val cases ({len(val_rows)} val)")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    val_rows = [r for r in rows if r["split"] == "val"]
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_by_case = {r["case_id"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    train_h = load_train_h()
    clean_val_h = load_clean_val_h()
    loaders = {"dirac": load_dirac, "cnn": load_cnn, "vit": load_vit}

    # one probe per architecture, trained ONCE on clean train h -- reused for every perturbation condition
    probes, clean_scores = {}, {}
    for arch in ("dirac", "cnn", "vit"):
        train_data = align(train_h[arch], ph_by_case)
        dim_in, dim_out = train_data["rep"].shape[1], train_data["target"].shape[1]
        probes[arch] = train_probe(dim_in, dim_out, train_data["rep"], train_data["target"], device)
        clean_val_data = align(clean_val_h[arch], ph_by_case)
        clean_scores[arch] = score(probes[arch], clean_val_data["rep"], clean_val_data["target"], device, SEED)
        log(f"{arch}: clean val macro_spearman={clean_scores[arch]['macro_spearman']:.4f}")

    results = {arch: {"clean_macro_spearman": clean_scores[arch]["macro_spearman"], "perturbations": {}}
              for arch in ("dirac", "cnn", "vit")}
    t0 = time.time()
    for cond_name, (kind, param) in PERTURBATIONS.items():
        for arch in ("dirac", "cnn", "vit"):
            model = loaders[arch]()
            perturbed_rep = extract_h_perturbed(arch, model, val_rows, device, kind, param)
            perturbed_data = align(perturbed_rep, ph_by_case)
            perturbed_score = score(probes[arch], perturbed_data["rep"], perturbed_data["target"], device, SEED)
            delta = clean_scores[arch]["macro_spearman"] - perturbed_score["macro_spearman"]
            results[arch]["perturbations"][cond_name] = {
                "perturbed_macro_spearman": perturbed_score["macro_spearman"], "degradation_delta": delta,
            }
            log(f"{cond_name}/{arch}: perturbed={perturbed_score['macro_spearman']:.4f} delta={delta:.4f} "
               f"(elapsed {time.time() - t0:.0f}s)")

    report = {
        "hypothesis": "H5 (robustness, experimental-spec Section 20 -- Section 21 distribution-shift is a separate, unaddressed gap, see module docstring)",
        "target": "compute_persistence_summary(X), 9 dims, log1p-transformed at the source -- same as H3-C, for continuity",
        "probe": "single Linear(dim_in, 9) per architecture, trained ONCE on clean train h, reused across every perturbation condition -- degradation is measured on the identical probe, not a re-fit one",
        "perturbations": {name: {"kind": kind, "param": param if not isinstance(param, tuple) else list(param)}
                          for name, (kind, param) in PERTURBATIONS.items()},
        "results": results,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / f"probe_h5_robustness_results_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_H5_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-h5] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
