# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 5 slice 2 (IMPLEMENTATION_SPEC.md): H3-C, the topology sub-hypothesis of H3 (Structural
Representation) -- experimental-spec's "does the latent recover topological properties T(X)?" -- run as a
two-condition design, not the naive single probe, because of a real confound found before any GPU spend:

DiracModel.forward requires `ph_vector` (the real, gudhi-computed T(X)) as an argument, and TopologyHead
embeds it directly into z via fusion (model/fusion.py). So `z -> T(X)` partially just asks "did the fusion
MLP preserve information it was handed directly," not "does DIRAC's structure help it perceive topology
from the image" -- CNN/ViT never receive ph_vector at all, so a naive z-based comparison would trivially
favor DIRAC by construction, not by architecture.

Two conditions, both frozen (no retraining, no architecture changes -- N_DIRAC-S stays 1,552,520):

  C0 -- full z (DIRAC only). DIRAC's official z, computed the normal way (ph_vector fed in). NOT the primary
  comparison -- a sanity check only: if AUROC/Spearman here is weak despite z having literally received the
  answer, something is wrong in the fusion pipeline. Never presented as evidence DIRAC "discovers" topology.

  C1 -- pre-fusion h, mean-pooled, for all three architectures. DIRAC: encoder.encode_hidden(x) (never sees
  ph_vector at inference). CNN: encode_hidden(x) (Phase 4), mean-pooled over the spatial grid. ViT:
  encode_hidden(x) (Phase 4), mean-pooled over tokens. This is the official H3-C comparison. Even so, it is
  NOT a clean "architecture alone" test: L_topology (jobs/pretrain_dirac_job.py) trains PHPredictHead on
  (psi, rho) against the real ph_vector as a stop-gradient target, and psi/rho trace back through the
  encoder -- so DIRAC's encoder received an explicit topology-prediction training signal that CNN/ViT's
  training recipe has no equivalent of at all (Phase 4 deliberately dropped L_struct/L_topology as having
  "no non-contrived CNN/ViT equivalent"). A DIRAC win at C1 could reflect the architecture, the L_topology
  training objective, or both -- C1 can't separate them. Disentangling that is a future ablation (train a
  DIRAC variant with L_topology removed entirely, not just the inference-time fusion injection), explicitly
  out of scope here per "do not modify DIRAC-S to improve H3."

Target T(X): the same, already-defined PH summary (model/topology.py's compute_persistence_summary -- no
new target invented), 9 dims = [count, sum, max] per homology dimension H0/H1/H2, already log1p-transformed
at the source (the same fix from Gate 3's smoke run, which found raw persistence stats run into the
thousands). Multi-output linear regression (Linear(dim_in, 9)), evaluated per-dimension via Spearman rank
correlation (primary, matching probe_linear_job.py's H1 regression diagnostic) and R^2, plus a macro-average
Spearman across the 9 dims. Same optimizer/epochs/seed/probe-architecture-type across all four conditions
(a single linear layer in every case) -- the only thing that legitimately differs across conditions is the
representation's native width (DIRAC h=64, DIRAC z=256, CNN h=288, ViT h=136), which is architecturally
unavoidable and not a violation of "same probe architecture."

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
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_h3c"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256          # DIRAC-S, locked
CNN_WIDTH, CNN_DEPTH = 36, 4                               # CNN3DBaseline, locked
VIT_WIDTH, VIT_DEPTH, VIT_HEADS, VIT_PATCH, VOLUME_SIZE = 136, 4, 4, 16, 128   # ViT3DBaseline, locked
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]
PH_DIM_NAMES = [f"H{d}_{s}" for d in range(TOPOLOGY_MAX_DIM + 1) for s in ("count", "sum", "max")]

SEED = int(os.environ.get("SEED", "0"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "300"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-2"))


def log(m):
    print(f"[probe-h3c] {m}", flush=True)


def load_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] in ("train", "val"))
    return rows


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


def feature_cache_path(condition: str, split: str) -> Path:
    return MNT / "processed" / "dirac" / "probe_features" / f"h3c_{condition}_{split}.npz"


def extract_dirac(rows: list[dict], device, ph_vectors: dict[str, np.ndarray]) -> None:
    """One pass, both conditions: h (C1, never touches ph_vector beyond the encoder call it never uses)
    and z (C0, ph_vector fed through topology_head+fusion) come from the same forward, avoiding a
    redundant second pass -- mirrors jobs/pretrain_dirac_job.py's encode_full pattern."""
    model = load_dirac().to(device)
    by_split: dict[str, list] = {"train": [], "val": []}
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            ph = torch.from_numpy(ph_vectors[row["local_path"]]).unsqueeze(0).to(device)

            h, _ = model.encoder.encode_hidden(x)                  # (1, N, d) -- C1, no ph_vector involved
            h_pooled = h.mean(dim=1)                                  # (1, d)
            psi = model.encoder.heads(h)
            rho = model.density_head(psi)
            graph = model.graph_head(psi)
            topology_embed = model.topology_head(ph)                    # C0's explicit side-channel
            z = model.fusion(rho, graph, topology_embed)                 # (1, z_dim) -- C0

            by_split[row["split"]].append((row["case_id"], h_pooled.squeeze(0).cpu().numpy(),
                                          z.squeeze(0).cpu().numpy()))
            if i % 50 == 0:
                log(f"dirac: extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        if not items:
            continue
        case_ids = np.array([it[0] for it in items])
        h_arr = np.stack([it[1] for it in items])
        z_arr = np.stack([it[2] for it in items])
        for condition, arr in (("dirac_h_c1", h_arr), ("dirac_z_c0", z_arr)):
            out_path = feature_cache_path(condition, split)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, rep=arr, case_id=case_ids)
        log(f"dirac/{split}: cached {len(items)} cases (h_c1 dim={h_arr.shape[1]}, z_c0 dim={z_arr.shape[1]})")


def extract_baseline(condition: str, model, rows: list[dict], device, pool_dims: tuple[int, ...]) -> None:
    model = model.to(device)
    by_split: dict[str, list] = {"train": [], "val": []}
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            out = model.encode_hidden(x)
            feat = out[0] if isinstance(out, tuple) else out           # ViT returns (tokens, grid_shape)
            pooled = feat.mean(dim=pool_dims)                             # (1, C)
            by_split[row["split"]].append((row["case_id"], pooled.squeeze(0).cpu().numpy()))
            if i % 50 == 0:
                log(f"{condition}: extracted {i}/{len(rows)}")

    for split, items in by_split.items():
        if not items:
            continue
        case_ids = np.array([it[0] for it in items])
        arr = np.stack([it[1] for it in items])
        out_path = feature_cache_path(condition, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, rep=arr, case_id=case_ids)
        log(f"{condition}/{split}: cached {len(items)} cases (dim={arr.shape[1]})")


def get_representations(condition: str, extractor, rows: list[dict], device, ph_vectors) -> dict[str, dict]:
    if feature_cache_path(condition, "train").exists() and feature_cache_path(condition, "val").exists():
        log(f"{condition}: using cached representations")
    else:
        extractor()
    out = {}
    for split in ("train", "val"):
        d = np.load(feature_cache_path(condition, split))
        out[split] = {"rep": d["rep"], "case_id": d["case_id"]}
    return out


def attach_targets(rep: dict[str, dict], ph_vectors_by_case: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for split, d in rep.items():
        keep = [i for i, cid in enumerate(d["case_id"]) if cid in ph_vectors_by_case]
        case_ids = d["case_id"][keep]
        out[split] = {
            "rep": d["rep"][keep], "case_id": case_ids,
            "target": np.stack([ph_vectors_by_case[cid] for cid in case_ids]),   # (N, 9)
        }
    return out


class LinearRegressionProbe(nn.Module):
    """Same probe architecture (a single linear layer) across all four conditions -- only its input width
    changes, which is architecturally unavoidable (each representation has a different native dimension),
    not a violation of "same probe architecture" across the comparison."""

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
    genuinely tiny but technically-nonzero variance can slip past evaluate_condition's own upstream
    degenerate-variance guard and then blow up here: ss_res stays a normal-sized number while ss_tot
    collapses toward zero, giving an astronomically negative "R^2" that looks like a real number but isn't
    one. Caught by a local dry run of the sibling probe_h3ab_geometry_relations_job.py before any GPU
    spend, not a hypothetical -- fixed here too since both files share this exact pattern."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else float("nan")


def evaluate_condition(name: str, data: dict[str, dict], device) -> dict:
    train, val = data["train"], data["val"]
    dim_in, dim_out = train["rep"].shape[1], train["target"].shape[1]
    probe = train_probe(dim_in, dim_out, train["rep"], train["target"], device)
    pred = predict(probe, val["rep"], device)

    per_dim = {}
    rhos = []
    for j, dim_name in enumerate(PH_DIM_NAMES):
        y_true, y_pred = val["target"][:, j], pred[:, j]
        if np.std(y_true) < 1e-4:   # degenerate target dim (e.g. no H2 features in any sampled case)
            per_dim[dim_name] = {"spearman_rho": None, "spearman_p": None, "r_squared": None, "note": "degenerate target (near-zero variance)"}
            continue
        sp = spearman_perm(y_pred, y_true, n=2000, seed=SEED)
        per_dim[dim_name] = {"spearman_rho": sp["rho"], "spearman_p": sp["p"], "r_squared": r_squared(y_true, y_pred)}
        rhos.append(sp["rho"])

    macro_spearman = float(np.mean(rhos)) if rhos else float("nan")
    log(f"{name}: dim_in={dim_in} macro_spearman={macro_spearman:.4f} "
       f"(n_dims_evaluated={len(rhos)}/{len(PH_DIM_NAMES)})")
    return {"dim_in": dim_in, "n_train": len(train["target"]), "n_val": len(val["target"]),
           "macro_spearman": macro_spearman, "per_dim": per_dim}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    rows = load_rows()
    log(f"{len(rows)} qc_passed train+val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_vectors = {r["local_path"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}
    ph_vectors_by_case = {r["case_id"]: ph_vectors[r["local_path"]] for r in ph_rows}

    t0 = time.time()
    cnn_model, vit_model = load_cnn(), load_vit()
    raw = {
        "dirac_h_c1": get_representations("dirac_h_c1", lambda: extract_dirac(rows, device, ph_vectors), rows, device, ph_vectors),
        "dirac_z_c0": get_representations("dirac_z_c0", lambda: extract_dirac(rows, device, ph_vectors), rows, device, ph_vectors),
        "cnn_h_c1": get_representations("cnn_h_c1", lambda: extract_baseline("cnn_h_c1", cnn_model, rows, device, pool_dims=(2, 3, 4)), rows, device, ph_vectors),
        "vit_h_c1": get_representations("vit_h_c1", lambda: extract_baseline("vit_h_c1", vit_model, rows, device, pool_dims=(1,)), rows, device, ph_vectors),
    }
    log(f"representation extraction/loading done in {time.time() - t0:.1f}s")
    data = {name: attach_targets(rep, ph_vectors_by_case) for name, rep in raw.items()}

    results = {name: evaluate_condition(name, d, device) for name, d in data.items()}

    report = {
        "hypothesis": "H3-C (topology sub-hypothesis of H3, Structural Representation)",
        "design": {
            "C0_dirac_z_c0": "DIRAC's full official z, ph_vector fed in via TopologyHead+fusion -- SANITY CHECK ONLY, not the primary comparison (DIRAC receives the answer as an input here)",
            "C1_official_comparison": ["dirac_h_c1", "cnn_h_c1", "vit_h_c1"],
            "caveat": "C1 is inference-time side-channel-free but NOT a clean architecture-alone test: DIRAC's encoder received an explicit L_topology training-time gradient (via PHPredictHead(psi,rho) against the real ph_vector) that CNN/ViT's training recipe has no equivalent of. A DIRAC C1 win could reflect the architecture, the L_topology objective, or both.",
        },
        "target": "compute_persistence_summary(X), 9 dims [count,sum,max] x H0/H1/H2, already log1p-transformed at the source",
        "probe": "single Linear(dim_in, 9) per condition, MSE loss, Adam lr=%.0e, %d epochs, seed=%d -- identical probe architecture/optimization/seed across all four conditions" % (PROBE_LR, PROBE_EPOCHS, SEED),
        "split_policy": "probe trained on train split, scored on val split; test split untouched (spec Section 24)",
        "results": results,
        "seeds": [SEED],
    }
    # seed-suffixed: this job's own SEED only randomizes the small probe head's init/optimization (the
    # cached representations above are deterministic given the frozen checkpoints, seed-independent) --
    # used to cheaply check probe-training-seed robustness on the SAME representations, distinct from (and
    # much cheaper than) re-running with independently retrained DIRAC/CNN/ViT checkpoints, which is
    # Phase 6's job. A shared, unsuffixed filename would let a later seed's run silently overwrite seed=0's.
    out_path = MNT / "processed" / "dirac" / f"probe_h3_topology_results_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_H3C_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-h3c] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
