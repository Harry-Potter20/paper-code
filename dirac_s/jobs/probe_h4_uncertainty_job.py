# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 5 slice 5 (IMPLEMENTATION_SPEC.md): H4, uncertainty -- dirac_formalization.md Section 8's
magnitude-conflation check, run for real for the first time (Section 8 was written specifically against a
documented failure in the sibling sheaf project: a disagreement signal validated on synthetic data but which
on real data INVERSELY predicted error, AUROC 0.138 vs. a plain baseline's 0.829, because it conflated
disagreement with token/signal magnitude rather than genuine uncertainty).

DIRAC-only, not a 3-way architecture comparison: CNN/ViT have no rho/entropy concept at all. The unit of
analysis is per-TOKEN (S(rho_i) is defined per token, formalization Section 3), not per-case -- 145 val
cases x 512 tokens = 74,240 data points, not 145.

Four checks, per Section 8:
  1. S(rho_i) vs downstream prediction error, via Spearman AND AUROC (error binarized above/below its own
     median) -- error = per-token L1 reconstruction error (the reconstruction objective already exists,
     model/losses.py's Decoder3D + jobs/pretrain_dirac_job.py's trained checkpoint; "reconstruction error"
     is explicitly one of Section 8's own suggested error signals, avoiding inventing a new one).
  2. Benchmark against a plain predictive-entropy baseline: the categorical Shannon entropy of the raw
     mixture weights H(pi_i) = -sum_k pi_k log(pi_k) (rho.weights, already computed, no basis/overlap
     geometry) vs S(rho_i)'s von Neumann entropy (which additionally accounts for how much the K basis
     vectors actually overlap, via the Gram-matrix trick in density_state.py). Tests whether the added
     complexity of the density-matrix formulation earns its keep over the much simpler comparator, playing
     the same role the sheaf project's "plain predictive-entropy baseline" played against its own
     disagreement signal D.
  3. Mandatory magnitude-conflation check: partial_spearman(S(rho), error, controlling_for=||psi||)
     (eval/stats.py, ported verbatim from the sheaf project's own H2b check, per Section 8's explicit "same
     statistic... for direct comparability across the two projects"). If this collapses toward zero relative
     to the raw (unpartialled) correlation, S(rho) is tracking token-norm magnitude, not genuine uncertainty
     -- the exact documented sheaf failure mode.
  4. Repeat (1) and (3) with mixture logits computed from L2-normalized (direction-only) psi_i -- a second,
     independent DensityHead forward pass on the SAME frozen weights, just with unit-norm input, no
     retraining. If entropy still discriminates confident from uncertain regions after norm is removed,
     that's positive evidence against the magnitude-proxy explanation.

No DIRAC-S modification: N_DIRAC-S stays 1,552,520. The decoder/ZBroadcast used here are the same
pretraining-only scaffolding jobs/pretrain_dirac_job.py trained and jobs/probe_h3_topology_job.py's sibling
jobs never touch -- loaded from the checkpoint, not retrained.

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
import torch.nn.functional as F

from model.complex_encoder import ComplexState
from model.dirac_model import DiracModel
from model.losses import Decoder3D, ZBroadcast
from data.local_cache import materialize_local, materialize_ph_vectors_local
from eval.stats import auroc, spearman_perm, partial_spearman

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_h4"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256
STEM_STRIDE = 16
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48
DATASETS = ["kits", "pancreas", "liver"]

SEED = int(os.environ.get("SEED", "0"))


def log(m):
    print(f"[probe-h4] {m}", flush=True)


def load_val_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.extend(r for r in ds_rows if r["qc_passed"] and r["split"] == "val")
    return rows


def load_checkpoint(device):
    ckpt = torch.load(MNT / "processed" / "dirac" / "checkpoints" / "pretrain_full_seed0.pt", map_location="cpu")
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=TOPOLOGY_MAX_DIM, z_dim=Z_DIM)
    model.load_state_dict(ckpt["model"])
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1))
    decoder.load_state_dict(ckpt["decoder"])
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D)
    z_broadcast.load_state_dict(ckpt["z_broadcast"])
    return model.eval().to(device), decoder.eval().to(device), z_broadcast.eval().to(device)


def categorical_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Plain Shannon entropy of the mixture weights alone -- H4's 'plain predictive-entropy baseline',
    playing the same comparator role the sheaf project's plain-entropy baseline played against its own
    disagreement signal. weights: (B, N, K) -> (B, N)."""
    w = weights.clamp_min(1e-12)
    return -(w * torch.log(w)).sum(-1)


def normalize_psi(psi: ComplexState) -> ComplexState:
    norm = psi.norm().unsqueeze(-1)
    return ComplexState(re=psi.re / norm, im=psi.im / norm)


def per_token_recon_error(recon: torch.Tensor, x: torch.Tensor, stem_stride: int) -> torch.Tensor:
    """L1 error, average-pooled from voxel space down to the token grid -- (B,1,D,H,W) -> (B, N)."""
    err = (recon - x).abs()
    pooled = F.avg_pool3d(err, kernel_size=stem_stride, stride=stem_stride)   # (B,1,dz,hy,wx)
    b = pooled.shape[0]
    return pooled.view(b, -1)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    rows = load_val_rows()
    log(f"{len(rows)} qc_passed val cases")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    ph_rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                           max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)
    ph_by_path = {r["local_path"]: np.load(r["local_ph_path"]).astype(np.float32) for r in ph_rows}

    model, decoder, z_broadcast = load_checkpoint(device)

    all_entropy, all_entropy_norm, all_categorical_h, all_norm_psi, all_error = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            vol = np.load(row["local_path"]).astype(np.float32)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            ph = torch.from_numpy(ph_by_path[row["local_path"]]).unsqueeze(0).to(device)

            h, grid_shape = model.encoder.encode_hidden(x)
            psi = model.encoder.heads(h)
            rho = model.density_head(psi)
            graph = model.graph_head(psi)
            topology_embed = model.topology_head(ph)
            z = model.fusion(rho, graph, topology_embed)
            decoder_in = z_broadcast(graph.node_features, z)
            recon = decoder(decoder_in, grid_shape)

            error = per_token_recon_error(recon, x, STEM_STRIDE)          # (1, N)
            entropy = rho.entropy()                                        # (1, N) -- S(rho_i)
            cat_h = categorical_entropy(rho.weights)                        # (1, N) -- plain baseline
            psi_norm_mag = psi.norm()                                       # (1, N) -- ||psi_i||

            psi_normalized = normalize_psi(psi)
            rho_norm = model.density_head(psi_normalized)   # same trained weights, unit-norm input
            entropy_norm = rho_norm.entropy()

            all_entropy.append(entropy.squeeze(0).cpu().numpy())
            all_entropy_norm.append(entropy_norm.squeeze(0).cpu().numpy())
            all_categorical_h.append(cat_h.squeeze(0).cpu().numpy())
            all_norm_psi.append(psi_norm_mag.squeeze(0).cpu().numpy())
            all_error.append(error.squeeze(0).cpu().numpy())

            if i % 20 == 0:
                log(f"processed {i}/{len(rows)}")

    entropy = np.concatenate(all_entropy)
    entropy_norm = np.concatenate(all_entropy_norm)
    categorical_h = np.concatenate(all_categorical_h)
    psi_norm = np.concatenate(all_norm_psi)
    error = np.concatenate(all_error)
    log(f"forward pass done in {time.time() - t0:.1f}s -- {len(entropy)} token-level data points")

    error_above_median = (error > np.median(error)).astype(int)

    def check_1_and_3(name: str, s: np.ndarray) -> dict:
        sp = spearman_perm(s, error, n=2000, seed=SEED)
        auc = auroc(s, error_above_median)
        pcorr = partial_spearman(s, error, controlling_for=psi_norm)
        return {"spearman_rho": sp["rho"], "spearman_p": sp["p"], "auroc_error_above_median": auc,
               "partial_spearman_given_psi_norm": pcorr,
               "collapses_toward_zero": abs(pcorr) < abs(sp["rho"]) * 0.5}

    result_entropy = check_1_and_3("S(rho)", entropy)
    result_categorical = check_1_and_3("H(pi) plain baseline", categorical_h)
    result_entropy_norm = check_1_and_3("S(rho) with L2-normalized psi", entropy_norm)

    report = {
        "hypothesis": "H4 (uncertainty, dirac_formalization.md Section 8 magnitude-conflation check)",
        "unit_of_analysis": f"per-token, not per-case: {len(rows)} val cases x 512 tokens = {len(entropy)} data points",
        "error_signal": "per-token L1 reconstruction error (Decoder3D on the frozen unmasked forward pass, loaded from the Phase-3 checkpoint's trained decoder/z_broadcast weights)",
        "check_1_S_rho_vs_error": result_entropy,
        "check_2_plain_baseline_H_pi_vs_error": result_categorical,
        "check_3_is_check_1s_partial_spearman_field_above": True,
        "check_4_S_rho_normalized_psi_vs_error": result_entropy_norm,
        "seeds": [SEED],
    }
    out_path = MNT / "processed" / "dirac" / "probe_h4_uncertainty_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE_H4_RESULT " + json.dumps(report), flush=True)
    print(f"[probe-h4] HALT — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
