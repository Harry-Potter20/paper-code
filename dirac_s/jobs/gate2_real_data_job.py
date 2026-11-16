# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Gate 2 (IMPLEMENTATION_SPEC.md Phase 2): re-run Gate 0's exact checks — rho PSD/trace-1, graph
connectivity, topology finite-ness, correct forward-pass shapes, no NaN/Inf anywhere — against REAL
preprocessed volumes from the frozen Gate-1 manifest, not synthetic tensors. Exists to catch real-data
edge cases (degenerate regions, HU outliers, near-empty foreground) before Phase 3's actual pretraining
spend, per the same before-expensive-training discipline as covtoken's own Gate 1.

Uses the locked Gate-0 architecture constants (d=64, K=8, configs/param_ledger.json) at the REAL target
scale (grid_shape=(8,8,8), 128^3 input, matching what Phase 3 pretraining will actually run) rather than
Gate 0's smaller synthetic-CI scale (4,4,4/64^3) — valid without recalibration because every layer in
DiracModel is per-token or fixed-kernel (conv/linear/GCN/transformer), so parameter count is provably
independent of grid_shape/N; only activation memory scales with N, not the parameter count Gate 0 matched.

Run with: -v hf://datasets/Chucks90/dirac-code:/dirac_pkg:ro -v hf://buckets/Chucks90/jobs-artifacts:/mnt
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/dirac_pkg")

import numpy as np
import torch
from scipy import ndimage

from model.dirac_model import DiracModel
from model.topology import compute_persistence_summary, TopologyHead

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
D, K = 64, 8   # locked in configs/param_ledger.json (Phase 0b) — do not change without a new Gate 0 run
GRID_SHAPE = (8, 8, 8)   # real target: 128^3 volume / 16 = 8^3 = 512 tokens
TOPOLOGY_DOWNSAMPLE = 48
N_SAMPLES_PER_DATASET = int(os.environ.get("N_SAMPLES_PER_DATASET", "12"))
DATASETS = ["kits", "pancreas", "liver"]


def log(m):
    print(f"[gate2] {m}", flush=True)


def sample_volume_paths() -> list[dict]:
    manifest_files = {
        "kits": MNT / "processed" / "dirac" / "kits_manifest.jsonl",
        "pancreas": MNT / "processed" / "dirac" / "pancreas_manifest.jsonl",
        "liver": MNT / "processed" / "dirac" / "liver_manifest.jsonl",
    }
    samples = []
    rng = random.Random(0)
    for ds, path in manifest_files.items():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        rows = [r for r in rows if r["qc_passed"]]
        chosen = rng.sample(rows, min(N_SAMPLES_PER_DATASET, len(rows)))
        for r in chosen:
            samples.append({"dataset": ds, "case_id": r["case_id"], "split": r["split"],
                            "out_path": r["out_path"]})
    return samples


def check_one(model: DiracModel, volume_path: Path) -> dict:
    vol = np.load(str(volume_path)).astype(np.float32)   # (128,128,128), already normalized to [-1,1]
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)    # (1,1,128,128,128)

    zoom = TOPOLOGY_DOWNSAMPLE / vol.shape[0]
    small = ndimage.zoom(vol, zoom, order=1)
    ph = compute_persistence_summary(small, max_dim=2)
    ph_vector = torch.from_numpy(ph).unsqueeze(0)

    with torch.no_grad():
        out = model(x, ph_vector=ph_vector)

    psd = out["rho"].check_psd_trace1()
    orphan = out["graph"].orphan_count()
    components = out["graph"].num_connected_components()
    topo_finite = bool(torch.isfinite(out["topology"]).all())
    z_finite = bool(torch.isfinite(out["z"]).all())
    psi_finite = bool(torch.isfinite(out["psi"].re).all() and torch.isfinite(out["psi"].im).all())

    return {
        "psd_ok": psd["psd_ok"], "min_eig": psd["min_eig"],
        "trace_ok": psd["trace_ok"], "max_trace_error": psd["max_trace_error"],
        "orphan_nodes": orphan, "connected_components": components,
        "topology_finite": topo_finite, "z_finite": z_finite, "psi_finite": psi_finite,
        "all_ok": (psd["psd_ok"] and psd["trace_ok"] and orphan == 0 and components == 1
                  and topo_finite and z_finite and psi_finite),
    }


def main():
    t0 = time.time()
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=2)
    model.eval()

    samples = sample_volume_paths()
    log(f"checking {len(samples)} real cases across {DATASETS}")

    results, failures = [], []
    for i, s in enumerate(samples):
        vpath = MNT / s["out_path"]
        try:
            check = check_one(model, vpath)
            results.append({**s, **check})
            if not check["all_ok"]:
                failures.append({**s, **check})
        except Exception as e:
            failures.append({**s, "error": f"{type(e).__name__}: {e}"})
        if i % 5 == 0:
            print("GATE2_PROGRESS " + json.dumps({"processed": i + 1, "failures": len(failures),
                                                   "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    n_ok = sum(1 for r in results if r.get("all_ok"))
    res = {
        "n_checked": len(samples), "n_ok": n_ok, "n_failures": len(failures),
        "failures": failures[:10],
        "by_dataset": {
            ds: {"checked": sum(1 for r in results if r["dataset"] == ds),
                "ok": sum(1 for r in results if r["dataset"] == ds and r.get("all_ok"))}
            for ds in DATASETS
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = MNT / "processed" / "dirac" / "gate2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("GATE2_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
