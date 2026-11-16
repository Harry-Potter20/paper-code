# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Gate 5 ablation disentangling probe (IMPLEMENTATION_SPEC.md Section 9 / experimental-spec Section 19):
reuses the H3-C topology probe (jobs/probe_h3_topology_job.py) and H2 data-efficiency curve
(jobs/probe_h2_data_efficiency_job.py) methodology verbatim -- same target (compute_persistence_summary,
9 dims), same probe (single Linear layer, MSE, Adam lr=1e-2, 300 epochs), same C1 extraction
(model.encoder.encode_hidden(x), mean-pooled over tokens -- NOT the fused z, which is DIRAC's C0
side-channel-injected condition per probe_h3_topology_job.py's own docstring, not something the ablations
need re-litigated) -- run against the 5 Phase-5 ablation checkpoints (D_classical, D_-topology, D_-complex,
D_-graph, D_-prob) instead of D_full/CNN/ViT, to answer the actual question the ablations exist for: does
each ablated model's H3-C macro-Spearman and H2 AULC collapse toward the CNN/ViT baselines, or hold near
D_full's own numbers? (D_full/CNN/ViT numbers themselves are NOT recomputed here -- they're already frozen
in gate_reports/gate_5_h3c_topology.json and gate_5_h2_data_efficiency.json; the comparison against them
happens locally after this job returns, not inside it.)

All 5 ablations share model.encoder.encode_hidden(x) completely unmodified -- none of the 5 touch the
shared stem+transformer, only downstream branches (rho/graph/topology/psi.im, or all four at once for
D_classical) -- so the SAME extraction code applies uniformly to all 5 checkpoints; only the trained
encoder WEIGHTS differ per ablation (each backpropped through a different downstream loss configuration
during its own training job), not the extraction mechanism.

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
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_ablation_probe"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256   # same as D_full -- locked, all 5 ablations share it
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
PH_DIM_NAMES = [f"H{d}_{s}" for d in range(TOPOLOGY_MAX_DIM + 1) for s in ("count", "sum", "max")]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
ABLATIONS = ["classical", "topology", "complex", "graph", "prob"]

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-ablations] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def load_ablation_model(name: str) -> DiracModel:
    ckpt_path = MNT / "processed" / "dirac" / "checkpoints" / f"ablation_{name}_full_seed0.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def feature_cache_path(name: str, split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"ablation_{name}_h_c1_{split}.npz"


def extract_ablation(name: str, rows: list[dict], device) -> None:
    model = load_ablation_model(name).to(device)
    by_split: dict[str, list] = {"train": [], "val": []}
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            h, _ = model.encoder.encode_hidden(x)      # identical across all 5 ablations and D_full
            h_pooled = h.mean(dim=1)
            by_split[row["split"]].append((row["case_id"], h_pooled.squeeze(0).cpu().numpy()))
            if i % 50 == 0:
                log(f"{name}: extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        if not items:
            continue
        case_ids = np.array([it[0] for it in items])
        arr = np.stack([it[1] for it in items])
        out_path = feature_cache_path(name, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, rep=arr, case_id=case_ids)
        log(f"{name}/{split}: cached {len(items)} cases (dim={arr.shape[1]})")


def get_h(name: str, rows: list[dict], device) -> dict[str, dict]:
    if feature_cache_path(name, "train").exists() and feature_cache_path(name, "val").exists():
        log(f"{name}: using cached representations")
    else:
        extract_ablation(name, rows, device)
    out = {}
    for split in ("train", "val"):
        d = np.load(feature_cache_path(name, split))
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


def evaluate_full(name: str, data: dict[str, dict], device) -> dict:
    train, val = data["train"], data["val"]
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    probe = train_probe(dim_in, dim_out, train["rep"], train["target"], device)
    pred = predict(probe, val["rep"], device)

    per_dim = {}
    rhos = []
    for j, dim_name in enumerate(PH_DIM_NAMES):
        y_true, y_pred = val["target"][:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            per_dim[dim_name] = {"spearman_rho": None, "spearman_p": None, "r_squared": None, "note": "degenerate target (near-zero variance)"}
            continue
        sp = spearman_perm(y_pred, y_true, n=2000, seed=SEED)
        per_dim[dim_name] = {"spearman_rho": sp["rho"], "spearman_p": sp["p"], "r_squared": r_squared(y_true, y_pred)}
        rhos.append(sp["rho"])

    macro = float(np.mean(rhos)) if rhos else float("nan")
    log(f"{name}: dim_in={dim_in} macro_spearman={macro:.4f} (n_dims_evaluated={len(rhos)}/{len(PH_DIM_NAMES)})")
    return {"dim_in": dim_in, "n_train": len(train["target"]), "n_val": len(val["target"]),
           "macro_spearman": macro, "per_dim": per_dim}


def evaluate_at_fraction(dim_in: int, dim_out: int, x_train: np.ndarray, y_train: np.ndarray,
                         x_val: np.ndarray, y_val: np.ndarray, device) -> dict:
    probe = train_probe(dim_in, dim_out, x_train, y_train, device)
    pred = predict(probe, x_val, device)
    rhos = []
    for j in range(dim_out):
        y_true, y_pred = y_val[:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=1000, seed=SEED)
        rhos.append(sp["rho"])
    return {"macro_spearman": float(np.mean(rhos)) if rhos else float("nan"), "n_dims_evaluated": len(rhos)}


def trapezoidal_area(y: list[float], x: list[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def nested_fraction_indices(n: int, fractions: list[float], seed: int) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out = {}
    for frac in fractions:
        k = max(1, int(round(n * frac)))
        out[frac] = order[:k]
    return out


def h2_curve(name: str, data: dict[str, dict], device) -> dict:
    train, val = data["train"], data["val"]
    n_train = len(train["target"])
    fraction_indices = nested_fraction_indices(n_train, FRACTIONS, SEED)
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]

    curve = []
    for frac in FRACTIONS:
        idx = fraction_indices[frac]
        metrics = evaluate_at_fraction(dim_in, dim_out, train["rep"][idx], train["target"][idx],
                                       val["rep"], val["target"], device)
        curve.append({"fraction": frac, "n_train_used": len(idx), **metrics})
        log(f"{name} @ {frac:.0%} (n={len(idx)}): macro_spearman={metrics['macro_spearman']:.4f}")

    finite_pts = [pt for pt in curve if np.isfinite(pt["macro_spearman"])]
    aulc = trapezoidal_area([pt["macro_spearman"] for pt in finite_pts],
                            [pt["fraction"] for pt in finite_pts]) if len(finite_pts) > 1 else float("nan")
    log(f"{name}: AULC={aulc:.4f}")
    return {"dim_in": dim_in, "curve": curve, "area_under_learning_curve": aulc}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_by_case = {r["case_id"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    h3c_results, h2_results = {}, {}
    for name in ABLATIONS:
        t0 = time.time()
        rep = get_h(name, rows, device)
        data = attach_targets(rep, ph_by_case)
        log(f"{name}: representation ready in {time.time() - t0:.1f}s")
        h3c_results[name] = evaluate_full(name, data, device)
        h2_results[name] = h2_curve(name, data, device)

    report = {
        "purpose": "Gate 5 ablation disentangling: re-runs the H3-C topology probe and H2 data-efficiency "
                  "curve against each of the 5 Phase-5 ablation checkpoints (same C1 extraction, same "
                  "target, same probe architecture as gate_5_h3c_topology.json / gate_5_h2_data_efficiency.json) "
                  "to test whether each ablated model's advantage over CNN/ViT collapses or survives. "
                  "D_full/CNN/ViT numbers are NOT recomputed here -- already frozen in those gate reports; "
                  "compared locally after this job returns.",
        "extraction": "model.encoder.encode_hidden(x), mean-pooled over tokens -- identical across all 5 "
                      "ablations and D_full, since none of the ablations modify the shared stem+transformer.",
        "target": "compute_persistence_summary(X), 9 dims, log1p-transformed at the source -- same as H3-C/H2",
        "probe": "single Linear(dim_in, 9), MSE, Adam lr=%.0e, %d epochs, seed=%d" % (PROBE_LR, PROBE_EPOCHS, SEED),
        "h3c_style_results": h3c_results,
        "h2_style_results": h2_results,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / f"ablations_probe_results_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_ABLATIONS_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-ablations] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
