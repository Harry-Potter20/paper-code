# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 6 (IMPLEMENTATION_SPEC.md): re-runs the H3-C-style topology probe and H2-style data-efficiency
curve (same target/probe methodology as gate_5_h3c_topology.json / gate_5_h2_data_efficiency.json /
jobs/probe_ablations_job.py) against INDEPENDENTLY-RETRAINED BACKBONE SEEDS, not just re-seeded probe heads
on a single frozen checkpoint -- the thing Gate 5 explicitly left for Phase 6 ("reproducibility across
independently-retrained backbone seeds ... that's Phase 6's job per spec, correctly deferred", CLAUDE.md).

Four architectures, three backbone seeds each (0/1/2, all already trained and PASSing at the training-
stability level): D_full (dirac), CNN-baseline, ViT-baseline, and D_-topology (the decisive Gate-5 ablation
-- its collapse-to-baseline finding is this whole spec's central claim and the one result Phase 6 most
needs to verify isn't a training-seed fluke). D_classical/D_-complex/D_-graph/D_-prob are NOT re-run here --
only seed=0 checkpoints exist for them; re-seeding those is a real follow-up but not part of this job
(scope: verify the paper's central claim first).

Extraction is identical across all four architectures' own encode_hidden-style methods (dirac/topology-
ablation share model.encoder.encode_hidden(x); CNN/ViT baselines have their own encode_hidden), mean-pooled
-- same C1 pattern probe_h3_topology_job.py and probe_ablations_job.py both already use.

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

from model.dirac_model import DiracModel
from baselines.cnn3d import CNN3DBaseline
from baselines.vit3d import ViT3DBaseline
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_multiseed"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256
CNN_WIDTH, CNN_DEPTH = 36, 4
VIT_WIDTH, VIT_DEPTH, VIT_HEADS, VIT_PATCH, VOLUME_SIZE = 136, 4, 4, 16, 128
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
PH_DIM_NAMES = [f"H{d}_{s}" for d in range(TOPOLOGY_MAX_DIM + 1) for s in ("count", "sum", "max")]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ARCHS = os.environ.get("ARCHS", "dirac,cnn,vit,topology").split(",")
OUT_TAG = os.environ.get("OUT_TAG", "")   # R1: write to probe_multiseed_results{OUT_TAG}.json, never over Gate 6

PROBE_SEED = int(os.environ.get("PROBE_SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-multiseed] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def checkpoint_path(arch: str, seed: int) -> Path:
    name = {"dirac": f"pretrain_full_seed{seed}.pt", "cnn": f"baseline_cnn_full_seed{seed}.pt",
           "vit": f"baseline_vit_full_seed{seed}.pt", "topology": f"ablation_topology_full_seed{seed}.pt",
           "cnn_topo": f"baseline_cnn_topo_full_seed{seed}.pt", "vit_topo": f"baseline_vit_topo_full_seed{seed}.pt"}[arch]
    return MNT / "processed" / "dirac" / "checkpoints" / name


def load_model(arch: str, seed: int):
    ckpt = torch.load(checkpoint_path(arch, seed), map_location="cpu")
    if arch in ("dirac", "topology"):
        model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    elif arch in ("cnn", "cnn_topo"):
        model = CNN3DBaseline(width=CNN_WIDTH, depth=CNN_DEPTH, z_dim=Z_DIM)
    elif arch in ("vit", "vit_topo"):
        model = ViT3DBaseline(width=VIT_WIDTH, depth=VIT_DEPTH, heads=VIT_HEADS, patch=VIT_PATCH,
                              volume_size=VOLUME_SIZE, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def feature_cache_path(arch: str, seed: int, split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"multiseed_{arch}_seed{seed}_h_{split}.npz"


def extract(arch: str, seed: int, rows: list[dict], device) -> None:
    model = load_model(arch, seed).to(device)
    by_split: dict[str, list] = {"train": [], "val": []}
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            if arch in ("dirac", "topology"):
                h, _ = model.encoder.encode_hidden(x)
            else:
                out = model.encode_hidden(x)
                h = out[0] if isinstance(out, tuple) else out
            pool_dims = (1,) if arch in ("dirac", "topology", "vit", "vit_topo") else (2, 3, 4)
            h_pooled = h.mean(dim=pool_dims)
            by_split[row["split"]].append((row["case_id"], h_pooled.squeeze(0).cpu().numpy()))
            if i % 100 == 0:
                log(f"{arch}/seed{seed}: extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        if not items:
            continue
        case_ids = np.array([it[0] for it in items])
        arr = np.stack([it[1] for it in items])
        out_path = feature_cache_path(arch, seed, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, rep=arr, case_id=case_ids)
        log(f"{arch}/seed{seed}/{split}: cached {len(items)} cases (dim={arr.shape[1]})")


def get_h(arch: str, seed: int, rows, device) -> dict[str, dict]:
    if feature_cache_path(arch, seed, "train").exists() and feature_cache_path(arch, seed, "val").exists():
        log(f"{arch}/seed{seed}: using cached representations")
    else:
        extract(arch, seed, rows, device)
    out = {}
    for split in ("train", "val"):
        d = np.load(feature_cache_path(arch, seed, split))
        out[split] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


def attach_targets(rep: dict[str, dict], ph_by_case: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for split, d in rep.items():
        keep = [i for i, cid in enumerate(d["case_id"]) if cid in ph_by_case]
        case_ids = d["case_id"][keep]
        out[split] = {"rep": d["rep"][keep], "case_id": case_ids,
                     "target": np.stack([ph_by_case[cid] for cid in case_ids])}
    return out


class LinearRegressionProbe(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)


def train_probe(dim_in: int, dim_out: int, x_train: np.ndarray, y_train: np.ndarray, device) -> nn.Module:
    torch.manual_seed(PROBE_SEED)
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
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else float("nan")


def evaluate_full(data: dict[str, dict], device) -> float:
    train, val = data["train"], data["val"]
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    probe = train_probe(dim_in, dim_out, train["rep"], train["target"], device)
    pred = predict(probe, val["rep"], device)
    rhos = []
    for j in range(dim_out):
        y_true, y_pred = val["target"][:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=2000, seed=PROBE_SEED)
        rhos.append(sp["rho"])
    return float(np.mean(rhos)) if rhos else float("nan")


def evaluate_at_fraction(dim_in, dim_out, x_train, y_train, x_val, y_val, device) -> float:
    probe = train_probe(dim_in, dim_out, x_train, y_train, device)
    pred = predict(probe, x_val, device)
    rhos = []
    for j in range(dim_out):
        y_true, y_pred = y_val[:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=1000, seed=PROBE_SEED)
        rhos.append(sp["rho"])
    return float(np.mean(rhos)) if rhos else float("nan")


def trapezoidal_area(y: list[float], x: list[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def nested_fraction_indices(n: int, fractions: list[float], seed: int) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return {frac: order[:max(1, int(round(n * frac)))] for frac in fractions}


def h2_aulc(data: dict[str, dict], device) -> float:
    train, val = data["train"], data["val"]
    n_train = len(train["target"])
    fraction_indices = nested_fraction_indices(n_train, FRACTIONS, PROBE_SEED)
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    curve = []
    for frac in FRACTIONS:
        idx = fraction_indices[frac]
        rho = evaluate_at_fraction(dim_in, dim_out, train["rep"][idx], train["target"][idx],
                                   val["rep"], val["target"], device)
        curve.append((frac, rho))
    finite = [(f, r) for f, r in curve if np.isfinite(r)]
    return trapezoidal_area([r for _, r in finite], [f for f, _ in finite]) if len(finite) > 1 else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} probe_seed={PROBE_SEED}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_by_case = {r["case_id"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    h3c_by_arch_seed, h2_by_arch_seed = {}, {}
    for arch in ARCHS:
        for seed in SEEDS:
            t0 = time.time()
            rep = get_h(arch, seed, rows, device)
            data = attach_targets(rep, ph_by_case)
            log(f"{arch}/seed{seed}: representation ready in {time.time() - t0:.1f}s")
            h3c_by_arch_seed[f"{arch}_seed{seed}"] = evaluate_full(data, device)
            h2_by_arch_seed[f"{arch}_seed{seed}"] = h2_aulc(data, device)
            log(f"{arch}/seed{seed}: h3c_macro_spearman={h3c_by_arch_seed[f'{arch}_seed{seed}']:.4f} "
               f"h2_aulc={h2_by_arch_seed[f'{arch}_seed{seed}']:.4f}")

    summary = {}
    for arch in ARCHS:
        h3c_vals = [h3c_by_arch_seed[f"{arch}_seed{s}"] for s in SEEDS]
        h2_vals = [h2_by_arch_seed[f"{arch}_seed{s}"] for s in SEEDS]
        summary[arch] = {
            "h3c_macro_spearman": {"by_seed": h3c_vals, "mean": float(np.mean(h3c_vals)), "std": float(np.std(h3c_vals))},
            "h2_aulc": {"by_seed": h2_vals, "mean": float(np.mean(h2_vals)), "std": float(np.std(h2_vals))},
        }

    report = {
        "purpose": "Phase 6: backbone-seed reproducibility of Gate 5's central disentangling finding "
                  "(D_-topology collapses to baseline; D_full/CNN/ViT margins). Re-runs H3-C/H2 methodology "
                  "against independently-retrained seed=0/1/2 checkpoints for dirac (D_full), cnn, vit, and "
                  "the D_-topology ablation -- NOT just re-seeded probe heads on one frozen checkpoint "
                  "(that was already checked in Gate 5).",
        "seeds": SEEDS, "probe_seed": PROBE_SEED,
        "per_seed_raw": {"h3c_macro_spearman": h3c_by_arch_seed, "h2_aulc": h2_by_arch_seed},
        "summary": summary,
    }
    out_path = MNT / "processed" / "dirac" / f"probe_multiseed_results{OUT_TAG}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_MULTISEED_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-multiseed] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
