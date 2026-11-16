# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 5 slice 4 (IMPLEMENTATION_SPEC.md): H2, data efficiency. Experimental-spec Section 22: "Downstream
labels should be evaluated at predetermined fractions: 1%, 5%, 10%, 25%, 50%, 100%. Does DIRAC retain useful
performance when labels are scarce? Area under the learning curve may be reported as a secondary summary."

Task choice: the topology probe (H3-C), not H1's tumor-size probe. Neither spec document names which
downstream task H2 should use, and this project has now run probes against several targets -- tumor-size
(H1, a robust null across the board even at 100% labels) and topology/geometry/relations (H3). A
label-fraction curve is only informative where there's a real signal to potentially lose as labels shrink;
H1's task has none even at full data, so re-running it under scarcity would mostly just show "stays near
chance regardless of fraction." H3-C's topology probe is the one target with a real, seed-robust DIRAC
advantage (gate_reports/gate_5_h3c_topology.json) -- testing whether that advantage holds, grows, or shrinks
under label scarcity is the scientifically meaningful question, and directly extends H3-C rather than
re-testing an already-null task.

Reuses H3-C's cached pre-fusion h (h3c_{dirac,cnn,vit}_h_c1_{split}.npz) and the same T(X) target
(compute_persistence_summary, 9 dims, log1p-transformed) -- no new GPU extraction, no new target invented.

Label fractions applied to the TRAIN split only (val stays fixed and full, for a fair per-fraction
comparison): 1%, 5%, 10%, 25%, 50%, 100%, via NESTED subsampling (a fixed seeded shuffle of train, then
prefixes of increasing length) rather than independent per-fraction resamples -- standard practice for
learning curves, since it means the 5% subset is a strict subset of the 10% subset, etc., reducing noise
from unrelated case composition changes between fractions (the curve reflects "more of the same base
sample," not "a different random sample entirely"). At 1% (~5 train cases for n_train=522), the CNN probe
(288-dim input) is wildly underdetermined -- reported anyway, since collapsing under extreme scarcity is
itself a real, informative data point about that architecture's representation, not a fitting failure to
paper over.

Same probe architecture/optimization/evaluation as H3-C (single Linear(dim_in, 9), MSE, Adam lr=1e-2, 300
epochs, per-dimension Spearman + macro-average) at every fraction, for direct comparability across the curve.

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

from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import spearman_perm

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_h2"))
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
PH_DIM_NAMES = [f"H{d}_{s}" for d in range(TOPOLOGY_MAX_DIM + 1) for s in ("count", "sum", "max")]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-h2] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


def h3c_cache_path(condition: str, split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"h3c_{condition}_h_c1_{split}.npz"


def load_h_representations() -> dict[str, dict[str, dict]]:
    out = {}
    for arch in ("dirac", "cnn", "vit"):
        out[arch] = {}
        for split in ("train", "val"):
            path = h3c_cache_path(arch, split)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run jobs/probe_h3_topology_job.py first.")
            d = np.load(path)
            out[arch][split] = {"rep": d["rep"], "case_id": d["case_id"]}
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


def evaluate_at_fraction(dim_in: int, dim_out: int, x_train: np.ndarray, y_train: np.ndarray,
                         x_val: np.ndarray, y_val: np.ndarray, device) -> dict:
    probe = train_probe(dim_in, dim_out, x_train, y_train, device)
    pred = predict(probe, x_val, device)
    rhos = []
    for j, dim_name in enumerate(PH_DIM_NAMES):
        y_true, y_pred = y_val[:, j], pred[:, j]
        if np.std(y_true) < 1e-4:
            continue
        sp = spearman_perm(y_pred, y_true, n=1000, seed=SEED)   # fewer perms: this runs 6x per architecture
        rhos.append(sp["rho"])
    return {"macro_spearman": float(np.mean(rhos)) if rhos else float("nan"), "n_dims_evaluated": len(rhos)}


def trapezoidal_area(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal rule, not np.trapz/np.trapezoid -- the function was renamed between NumPy 1.x
    and 2.x (trapz removed in 2.x in favor of trapezoid), and pinning a numpy version just to avoid that
    is more fragile than five lines of arithmetic. Caught by a local dry run, not a hypothetical."""
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def nested_fraction_indices(n: int, fractions: list[float], seed: int) -> dict[float, np.ndarray]:
    """A single seeded shuffle, then prefixes -- so the 5% subset is a strict subset of the 10% subset,
    etc. (nested), rather than independent resamples per fraction. Standard for learning curves: isolates
    "more of the same base sample" from noise due to an unrelated case-composition change between points."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out = {}
    for frac in fractions:
        k = max(1, int(round(n * frac)))
        out[frac] = order[:k]
    return out


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

    t0 = time.time()
    h_rep = load_h_representations()
    log(f"loaded cached h representations in {time.time() - t0:.1f}s")
    data = {arch: attach_targets(rep, ph_by_case) for arch, rep in h_rep.items()}

    results = {}
    for arch, d in data.items():
        train, val = d["train"], d["val"]
        n_train = len(train["target"])
        fraction_indices = nested_fraction_indices(n_train, FRACTIONS, SEED)
        dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]

        curve = []
        for frac in FRACTIONS:
            idx = fraction_indices[frac]
            metrics = evaluate_at_fraction(dim_in, dim_out, train["rep"][idx], train["target"][idx],
                                           val["rep"], val["target"], device)
            curve.append({"fraction": frac, "n_train_used": len(idx), **metrics})
            log(f"{arch} @ {frac:.0%} (n={len(idx)}): macro_spearman={metrics['macro_spearman']:.4f}")

        finite_pts = [pt for pt in curve if np.isfinite(pt["macro_spearman"])]
        aulc = trapezoidal_area([pt["macro_spearman"] for pt in finite_pts],
                                [pt["fraction"] for pt in finite_pts]) if len(finite_pts) > 1 else float("nan")
        results[arch] = {"dim_in": dim_in, "curve": curve, "area_under_learning_curve": aulc}
        log(f"{arch}: AULC={aulc:.4f}")

    report = {
        "hypothesis": "H2 (data efficiency)",
        "task_choice_rationale": "topology probe (H3-C's target and representations), not H1's tumor-size probe -- see module docstring: H1 is a robust null even at 100% labels, so a scarcity curve on it wouldn't be informative; H3-C has a real, seed-robust DIRAC advantage worth stress-testing under scarcity.",
        "fractions": FRACTIONS,
        "sampling": "nested (single seeded shuffle of train, then prefixes) -- not independent per-fraction resamples",
        "target": "compute_persistence_summary(X), 9 dims, log1p-transformed at the source -- same as H3-C",
        "probe": "single Linear(dim_in, 9) per architecture, MSE loss, Adam lr=%.0e, %d epochs, seed=%d, re-fit independently at each fraction" % (PROBE_LR, PROBE_EPOCHS, SEED),
        "split_policy": "val split (n=145) fixed and full at every fraction; only the train subset used to fit the probe shrinks; test split untouched (spec Section 24)",
        "results": results,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / f"probe_h2_data_efficiency_results_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_H2_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-h2] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
