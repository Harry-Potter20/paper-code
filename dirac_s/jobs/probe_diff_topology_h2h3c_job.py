# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Diff-topology follow-on (dirac_diff_topology_formalization.md): the H2/H3-C half of the "next" step
gate_reports/gate_diff_topology_smoke_lamtd0p01.json names ("rerun the H2/H3-C probe methodology against
this checkpoint vs. D_full's"). The retrain half is already done -- pretrain_diff_topology_full_seed0_lamtd0p01.pt,
same seed/data/steps as D_full's own full run (this job's checkpoint loader, extraction, and probe
methodology are otherwise IDENTICAL to jobs/probe_h3_topology_job.py's C1 condition and
jobs/probe_h2_data_efficiency_job.py -- same target, same probe architecture/optimizer/epochs, same 3
probe-training seeds -- so this job's numbers are directly comparable to gate_5_h3c_topology.json's
C1_dirac_h and gate_5_h2_data_efficiency.json's dirac AULC without re-deriving either.

D_full's own numbers are NOT recomputed here -- reused verbatim from the two gate files above (DFULL_H3C_C1,
DFULL_H2_AULC below) -- only the new checkpoint needs fresh extraction+probing.

Model architecture is DiracModel, unmodified -- the diff-topology training run never touched
model/dirac_model.py (dirac_diff_topology_formalization.md Section 4: "No modification to
model/dirac_model.py... this experiment's code lives entirely in new files"), so this job's checkpoint
loader and encode_hidden extraction are byte-identical to every other DIRAC-S probe job's, not a new path.

Run with: -v hf://datasets/Chucks90/dirac-code:/dirac_pkg:ro -v hf://buckets/Chucks90/jobs-artifacts:/mnt
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/dirac_pkg")

import numpy as np
import torch
import torch.nn as nn

from model.dirac_model import DiracModel
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_difftopo_h2h3c"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256  # DIRAC-S, locked -- same as every other probe job
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
PROBE_SEEDS = [0, 1, 2]  # this vault's convention throughout Gate 5/6: 3 probe-training seeds per finding
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))
LAM_TD_TAG = os.environ.get("LAM_TD_TAG", "0p01")
CKPT_NAME = f"pretrain_diff_topology_full_seed0_lamtd{LAM_TD_TAG}.pt"

# D_full's already-recorded, signed-off numbers (gate_reports/gate_5_h3c_topology.json /
# gate_5_h2_data_efficiency.json) -- compared against directly, not recomputed.
DFULL_H3C_C1 = {"by_probe_seed": [0.6917396109978476, 0.7171881227116481, 0.7249440852992478], "mean": 0.7113}
DFULL_H2_AULC = {"by_probe_seed": [0.7164437941118749, 0.6966483439563269, 0.7029], "mean": 0.7053}


def log(m):
    print(f"[probe-difftopo-h2h3c] {m}", flush=True)


def load_rows() -> list[dict]:
    rows = []
    for ds in DATASETS:
        p = MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl"
        ds_rows = [json.loads(l) for l in open(p) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def load_model():
    ckpt_path = MNT / "processed" / "dirac" / "checkpoints" / CKPT_NAME
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt_path


def feature_cache_path(split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"h3c_difftopo_lamtd{LAM_TD_TAG}_h_c1_{split}.npz"


def extract(model, rows: list[dict], device) -> None:
    by_split: dict[str, list] = {"train": [], "val": []}
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            h, _ = model.encoder.encode_hidden(x)          # C1: pre-fusion h, same as gate_5_h3c_topology.json
            h_pooled = h.mean(dim=1)
            by_split[row["split"]].append((row["case_id"], h_pooled.squeeze(0).cpu().numpy()))
            if i % 100 == 0:
                log(f"extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        case_ids = np.array([it[0] for it in items])
        arr = np.stack([it[1] for it in items])
        out_path = feature_cache_path(split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, rep=arr, case_id=case_ids)
        log(f"{split}: cached {len(items)} cases (dim={arr.shape[1]})")


def get_h(model, rows, device) -> dict[str, dict]:
    if feature_cache_path("train").exists() and feature_cache_path("val").exists():
        log("using cached representations")
    else:
        extract(model, rows, device)
    out = {}
    for split in ("train", "val"):
        d = np.load(feature_cache_path(split))
        out[split] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


def attach_targets(rep, ph_by_case) -> dict[str, dict]:
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


def train_probe(dim_in, dim_out, x_train, y_train, device, probe_seed) -> nn.Module:
    torch.manual_seed(probe_seed)
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


def predict(probe, x, device) -> np.ndarray:
    with torch.no_grad():
        return probe(torch.from_numpy(x).float().to(device)).cpu().numpy()


def evaluate_full(data, device, probe_seed) -> float:
    train, val = data["train"], data["val"]
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    probe = train_probe(dim_in, dim_out, train["rep"], train["target"], device, probe_seed)
    pred = predict(probe, val["rep"], device)
    rhos = []
    for j in range(dim_out):
        y_true, y_pred = val["target"][:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=2000, seed=probe_seed)
        rhos.append(sp["rho"])
    return float(np.mean(rhos)) if rhos else float("nan")


def evaluate_at_fraction(dim_in, dim_out, x_train, y_train, x_val, y_val, device, probe_seed) -> float:
    probe = train_probe(dim_in, dim_out, x_train, y_train, device, probe_seed)
    pred = predict(probe, x_val, device)
    rhos = []
    for j in range(dim_out):
        y_true, y_pred = y_val[:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=1000, seed=probe_seed)
        rhos.append(sp["rho"])
    return float(np.mean(rhos)) if rhos else float("nan")


def trapezoidal_area(y, x) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def nested_fraction_indices(n, fractions, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return {frac: order[:max(1, int(round(n * frac)))] for frac in fractions}


def h2_aulc(data, device, probe_seed) -> float:
    train, val = data["train"], data["val"]
    n_train = len(train["target"])
    fraction_indices = nested_fraction_indices(n_train, FRACTIONS, probe_seed)
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    curve = []
    for frac in FRACTIONS:
        idx = fraction_indices[frac]
        rho = evaluate_at_fraction(dim_in, dim_out, train["rep"][idx], train["target"][idx],
                                   val["rep"], val["target"], device, probe_seed)
        curve.append((frac, rho))
    finite = [(f, r) for f, r in curve if np.isfinite(r)]
    return trapezoidal_area([r for _, r in finite], [f for f, _ in finite]) if len(finite) > 1 else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} ckpt={CKPT_NAME}")

    model, ckpt_path = load_model()
    model = model.to(device)
    log(f"loaded checkpoint {ckpt_path}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_by_case = {r["case_id"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    rep = get_h(model, rows, device)
    data = attach_targets(rep, ph_by_case)
    log(f"representation ready: train n={len(data['train']['case_id'])} val n={len(data['val']['case_id'])} "
        f"dim_in={data['train']['rep'].shape[1]}")

    h3c_by_seed, h2_by_seed = [], []
    for ps in PROBE_SEEDS:
        h3c = evaluate_full(data, device, ps)
        h2 = h2_aulc(data, device, ps)
        h3c_by_seed.append(h3c)
        h2_by_seed.append(h2)
        log(f"probe_seed={ps}: h3c_macro_spearman={h3c:.4f} h2_aulc={h2:.4f}")

    h3c_mean = float(np.mean(h3c_by_seed))
    h2_mean = float(np.mean(h2_by_seed))

    report = {
        "project": "dirac", "gate": "diff_topology_h2h3c_comparison",
        "phase": "Diff-topology follow-on - H2/H3-C comparison against D_full",
        "objective": "Does the differentiable topology-fidelity loss (LAM_TD=0.01, "
                     "gate_diff_topology_smoke_lamtd0p01.json PASS) change DIRAC-S's H2/H3-C story versus "
                     "D_full's own signed-off numbers?",
        "checkpoint": str(ckpt_path), "dim_in": int(data["train"]["rep"].shape[1]),
        "probe_seeds": PROBE_SEEDS,
        "target": "compute_persistence_summary(X), same 9-dim log1p target as gate_5_h3c_topology.json / "
                  "gate_5_h2_data_efficiency.json, not recomputed differently.",
        "extraction": "C1 pattern: encoder.encode_hidden(x) pre-fusion h, mean-pooled over tokens -- "
                      "identical to D_full's own C1_dirac_h extraction in gate_5_h3c_topology.json.",
        "checks": {
            "h3c_macro_spearman": {
                "difftopo_lamtd0p01": {"by_probe_seed": h3c_by_seed, "mean": h3c_mean},
                "d_full_c1_dirac_h": DFULL_H3C_C1,
                "delta_mean": h3c_mean - DFULL_H3C_C1["mean"],
            },
            "h2_aulc": {
                "difftopo_lamtd0p01": {"by_probe_seed": h2_by_seed, "mean": h2_mean},
                "d_full": DFULL_H2_AULC,
                "delta_mean": h2_mean - DFULL_H2_AULC["mean"],
            },
        },
    }
    out_path = MNT / "processed" / "dirac" / "probe_diff_topology_h2h3c_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_DIFFTOPO_H2H3C_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-difftopo-h2h3c] HALT -- results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
