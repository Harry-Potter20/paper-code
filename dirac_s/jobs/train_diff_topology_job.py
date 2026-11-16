# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""dirac_diff_topology_formalization.md Section 5's real-data smoke gate: does adding a differentiable
topology-fidelity loss on the decoder's own reconstruction, on top of D_full's existing recipe, converge
without instability -- same PASS bar pretrain_dirac_job.py's own Gate 3 used (0 NaN/Inf steps, loss
decreasing in the tail). This is NOT a Gate-6 file and does not touch gate_3.json, pretrain_dirac_job.py, or
either checkpoint path pretrain_dirac_job.py writes -- entirely new outputs (gate_diff_topology_*.json,
pretrain_diff_topology_*.pt), so a bad run here can never clobber the signed-off D_full artifacts.

**Additive, not a replacement** (formalization Section 2d): every one of D_full's four losses
(L_recon/L_masked/L_struct/L_topology via PHPredictHead) runs completely unmodified, verbatim from
pretrain_dirac_job.py -- this script is that job plus one new term, `loss_topology_diff`, not a rewrite of
it. If this run doesn't converge, the failure is attributable to the new term, not to a from-scratch
reimplementation bug.

**Self-contained addition, not a package-mount change**: pretrain_dirac_job.py mounts the frozen
`hf://datasets/Chucks90/dirac-code` package read-only for DiracModel/losses/etc -- reused verbatim here via
the same mount. `model/topology_diff.py` (new, tested via tests/test_topology_diff.py, never pushed to that
frozen dataset) is inlined directly below instead, so this experiment never needs to modify the package
every other already-signed-off job depends on.

**Real, flagged cost**: unlike L_topology's target (cached once per case via a ProcessPoolExecutor,
data/local_cache.py), `loss_topology_diff`'s gudhi call runs on the reconstruction, which changes every
step -- no caching is possible. Per-step wall time for this new term is printed separately in the progress
log (`gudhi_s`) so the real cost is measured on this run, not guessed.

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
import torch.nn as nn
import torch.nn.functional as F
import gudhi
from scipy import ndimage

from model.complex_encoder import ComplexState
from model.density_state import DensityState
from model.dirac_model import DiracModel
from model.topology import compute_persistence_summary, TopologyHead
from model.losses import (
    Decoder3D, PHPredictHead, ZBroadcast,
    generate_block_mask, upsample_token_mask_to_voxels,
    l_recon, l_masked, l_struct, l_topology, l_topology_stability, l_dirac,
)
from data.local_cache import materialize_local, materialize_ph_vectors_local


# ============================================================================
# Inlined from model/topology_diff.py (dirac_diff_topology_formalization.md Section 2a/2c) -- see that
# module's docstring for the live-verified gudhi indexing claim this relies on, and
# tests/test_topology_diff.py for the correctness/gradient checks. Duplicated here, not imported, per this
# file's module docstring.
# ============================================================================

def _pairing_indices(volume_np: np.ndarray, max_dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cc = gudhi.CubicalComplex(top_dimensional_cells=volume_np.flatten(), dimensions=list(volume_np.shape))
    cc.compute_persistence()
    coface_pairs = cc.cofaces_of_persistence_pairs()[0]
    out = []
    for dim in range(max_dim + 1):
        pairs_dim = coface_pairs[dim] if dim < len(coface_pairs) else np.zeros((0, 2), dtype=np.int64)
        out.append((pairs_dim[:, 0].astype(np.int64), pairs_dim[:, 1].astype(np.int64)))
    return out


def differentiable_persistence_summary(volume: torch.Tensor, max_dim: int = 2) -> torch.Tensor:
    assert volume.dim() == 3, f"expected (D,H,W), got {tuple(volume.shape)}"
    with torch.no_grad():
        neg_np = (-volume).detach().cpu().numpy().astype(np.float64)
        pairings = _pairing_indices(neg_np, max_dim)

    flat = (-volume).flatten()
    stats = []
    for birth_idx, death_idx in pairings:
        if len(birth_idx) == 0:
            stats.extend([flat.new_zeros(()), flat.new_zeros(()), flat.new_zeros(())])
            continue
        b_idx_t = torch.as_tensor(birth_idx, device=volume.device, dtype=torch.long)
        d_idx_t = torch.as_tensor(death_idx, device=volume.device, dtype=torch.long)
        lifetimes = flat[d_idx_t] - flat[b_idx_t]
        lifetimes = torch.clamp(lifetimes, min=0.0)
        count = flat.new_tensor(float(len(birth_idx)))
        stats.extend([count, lifetimes.sum(), lifetimes.max()])
    summary = torch.stack(stats)
    return torch.log1p(summary)


def differentiable_persistence_summary_batch(volumes: torch.Tensor, max_dim: int = 2) -> torch.Tensor:
    if volumes.dim() == 5:
        volumes = volumes.squeeze(1)
    assert volumes.dim() == 4, f"expected (B,D,H,W) or (B,1,D,H,W), got {tuple(volumes.shape)}"
    return torch.stack([differentiable_persistence_summary(v, max_dim=max_dim) for v in volumes])


# ============================================================================
# Everything below is pretrain_dirac_job.py verbatim except: (1) LAM_TD + loss_topology_diff (marked
# "NEW" at each site), (2) output paths (checkpoint_path, gate report name, log prefixes) renamed so
# nothing here can collide with pretrain_dirac_job.py's own outputs.
# ============================================================================

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache"))
D, K = 64, 8
GRID_SHAPE = (8, 8, 8)
STEM_STRIDE = 16
Z_DIM = 256
TOPOLOGY_MAX_DIM = 2
TOPOLOGY_DOWNSAMPLE = 48
DATASETS = ["kits", "pancreas", "liver"]

SMOKE = os.environ.get("SMOKE", "1") == "1"
N_SMOKE_CASES_PER_DATASET = int(os.environ.get("N_SMOKE_CASES_PER_DATASET", "16"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "3" if SMOKE else "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
LR = float(os.environ.get("LR", "1e-4"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
MASK_RATIO = float(os.environ.get("MASK_RATIO", "0.75"))
LAM_R = float(os.environ.get("LAM_R", "1.0"))
LAM_M = float(os.environ.get("LAM_M", "1.0"))
LAM_S = float(os.environ.get("LAM_S", "1.0"))
LAM_T = float(os.environ.get("LAM_T", "1.0"))
LAM_TD = float(os.environ.get("LAM_TD", "1.0"))   # NEW: weight on loss_topology_diff. [CALIBRATE], no prior data.
MAX_SHIFT_TOKENS = int(os.environ.get("MAX_SHIFT_TOKENS", "2"))
TOPOLOGY_STABILITY_EVERY = 10
SEED = int(os.environ.get("SEED", "0"))


def log(m):
    print(f"[diff-topology] {m}", flush=True)


def load_train_rows() -> list[dict]:
    manifest_files = {ds: MNT / "processed" / "dirac" / f"{ds}_manifest.jsonl" for ds in DATASETS}
    rng = random.Random(SEED)
    rows = []
    for ds, path in manifest_files.items():
        ds_rows = [json.loads(l) for l in open(path) if l.strip()]
        ds_rows = [r for r in ds_rows if r["qc_passed"] and r["split"] == "train"]
        if SMOKE:
            ds_rows = rng.sample(ds_rows, min(N_SMOKE_CASES_PER_DATASET, len(ds_rows)))
        rows.extend(ds_rows)
    return rows


def load_batch(rows: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    vols = np.stack([np.load(r["local_path"]).astype(np.float32) for r in rows])
    phs = np.stack([np.load(r["local_ph_path"]).astype(np.float32) for r in rows])
    x = torch.from_numpy(vols).unsqueeze(1).to(device)
    ph = torch.from_numpy(phs).to(device)
    return x, ph


def sample_structure_transform() -> tuple[tuple[int, int, int], float, float]:
    shift_tokens = tuple(random.randint(-MAX_SHIFT_TOKENS, MAX_SHIFT_TOKENS) for _ in range(3))
    jitter_scale = random.uniform(0.9, 1.1)
    jitter_offset = random.uniform(-0.05, 0.05)
    return shift_tokens, jitter_scale, jitter_offset


def apply_structure_transform(x: torch.Tensor, shift_tokens, jitter_scale: float,
                              jitter_offset: float) -> torch.Tensor:
    kz, ky, kx = shift_tokens
    sz, sy, sx = kz * STEM_STRIDE, ky * STEM_STRIDE, kx * STEM_STRIDE

    def shift_slices(s: int, size: int) -> tuple[slice, slice]:
        return (slice(0, size - s), slice(s, size)) if s >= 0 else (slice(-s, size), slice(0, size + s))

    src_z, dst_z = shift_slices(sz, x.shape[2])
    src_y, dst_y = shift_slices(sy, x.shape[3])
    src_x, dst_x = shift_slices(sx, x.shape[4])
    x_shifted = torch.zeros_like(x)
    x_shifted[:, :, dst_z, dst_y, dst_x] = x[:, :, src_z, src_y, src_x]
    return (x_shifted * jitter_scale + jitter_offset).clamp(-1.0, 1.0)


def matched_token_slices(grid_shape, shift_tokens):
    def axis_slices(k: int, size: int) -> tuple[slice, slice]:
        return (slice(k, size), slice(0, size - k)) if k >= 0 else (slice(0, size + k), slice(-k, size))
    dz, hy, wx = grid_shape
    kz, ky, kx = shift_tokens
    xz, gz = axis_slices(kz, dz)
    xy, gy = axis_slices(ky, hy)
    xx, gx = axis_slices(kx, wx)
    return (xz, xy, xx), (gz, gy, gx)


def crop_density_state(rho: DensityState, grid_shape, slices) -> DensityState:
    dz, hy, wx = grid_shape
    b, _, k = rho.weights.shape
    sz, sy, sx = slices
    w = rho.weights.view(b, dz, hy, wx, k)[:, sz, sy, sx, :].reshape(b, -1, k)
    d = rho.basis.re.shape[-1]
    re = rho.basis.re.view(b, dz, hy, wx, k, d)[:, sz, sy, sx, :, :].reshape(b, -1, k, d)
    im = rho.basis.im.view(b, dz, hy, wx, k, d)[:, sz, sy, sx, :, :].reshape(b, -1, k, d)
    return DensityState(weights=w, basis=ComplexState(re=re, im=im))


def encode_full(model: DiracModel, x: torch.Tensor, ph_vector: torch.Tensor):
    h, grid_shape = model.encoder.encode_hidden(x)
    psi = model.encoder.heads(h)
    rho = model.density_head(psi)
    graph = model.graph_head(psi)
    topology_embed = model.topology_head(ph_vector)
    z = model.fusion(rho, graph, topology_embed)
    return {"h": h, "grid_shape": grid_shape, "psi": psi, "rho": rho, "graph": graph,
            "topology_embed": topology_embed, "z": z}


def checkpoint_path() -> Path:
    # NEW filename ("diff_topology" infix) -- never the same path pretrain_dirac_job.py writes to.
    # lam_td is part of the filename (not just seed/smoke): LAM_TD is an explicitly [CALIBRATE] value with no
    # prior data (line ~134), so different LAM_TD runs are different experiments, not resumptions of each
    # other -- without this, a same-seed rerun at a new LAM_TD silently loads the old LAM_TD's finished
    # checkpoint, does zero new training, and re-emits its old metrics verbatim (real bug hit 2026-08-20:
    # a LAM_TD=0.01 rerun against an already-complete LAM_TD=1.0 checkpoint produced a byte-identical
    # DIFFTOPO_RESULT to 15 significant digits -- the giveaway that no new step ever ran).
    lam_td_tag = f"{LAM_TD:g}".replace(".", "p")
    return MNT / "processed" / "dirac" / "checkpoints" / f"pretrain_diff_topology_{'smoke' if SMOKE else 'full'}_seed{SEED}_lamtd{lam_td_tag}.pt"


def save_checkpoint(model, decoder, ph_predict_head, z_broadcast, optimizer, epoch, step, history, nan_inf_steps):
    ckpt_path = checkpoint_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(), "decoder": decoder.state_dict(),
        "ph_predict_head": ph_predict_head.state_dict(), "z_broadcast": z_broadcast.state_dict(),
        "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
        "history": history, "nan_inf_steps": nan_inf_steps,
    }, tmp)
    tmp.rename(ckpt_path)


def load_checkpoint_if_present(model, decoder, ph_predict_head, z_broadcast, optimizer, device):
    ckpt_path = checkpoint_path()
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    ph_predict_head.load_state_dict(ckpt["ph_predict_head"])
    z_broadcast.load_state_dict(ckpt["z_broadcast"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["step"], ckpt["history"], ckpt["nan_inf_steps"]


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} smoke={SMOKE} epochs={N_EPOCHS} batch_size={BATCH_SIZE} lam_td={LAM_TD}")

    rows = load_train_rows()
    log(f"materializing {len(rows)} train cases to local disk")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)
    rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache",
                                        max_dim=TOPOLOGY_MAX_DIM, downsample=TOPOLOGY_DOWNSAMPLE)

    m = TopologyHead.summary_length(TOPOLOGY_MAX_DIM)
    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, topology_max_dim=TOPOLOGY_MAX_DIM, z_dim=Z_DIM).to(device)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1)).to(device)
    ph_predict_head = PHPredictHead(d=D, K=K, m=m).to(device)
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D).to(device)

    params = list(model.parameters()) + list(decoder.parameters()) + \
             list(ph_predict_head.parameters()) + list(z_broadcast.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    history = {"step": [], "loss_total": [], "loss_recon": [], "loss_masked": [], "loss_struct": [],
              "loss_topology": [], "loss_topology_diff": [], "loss_topology_stability": [],
              "grad_norm": [], "gudhi_s": []}   # NEW: loss_topology_diff, gudhi_s
    nan_inf_steps = 0
    step = 0
    start_epoch = 0
    if not SMOKE and os.environ.get("RESUME", "1") == "1":
        resumed = load_checkpoint_if_present(model, decoder, ph_predict_head, z_broadcast, optimizer, device)
        if resumed is not None:
            start_epoch, step, history, nan_inf_steps = resumed
            start_epoch += 1
            log(f"resumed from checkpoint: epoch={start_epoch} step={step}")
    t0 = time.time()

    for epoch in range(start_epoch, N_EPOCHS):
        random.shuffle(rows)
        for i in range(0, len(rows) - len(rows) % BATCH_SIZE, BATCH_SIZE):
            batch_rows = rows[i:i + BATCH_SIZE]
            x, ph_vector = load_batch(batch_rows, device)

            out_x = encode_full(model, x, ph_vector)
            decoder_in = z_broadcast(out_x["graph"].node_features, out_x["z"])
            recon = decoder(decoder_in, out_x["grid_shape"])
            loss_recon = l_recon(recon, x)

            token_mask = generate_block_mask(out_x["grid_shape"], mask_ratio=MASK_RATIO,
                                             batch=x.shape[0], device=device)
            voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
            masked_x = x.masked_fill(voxel_mask, 0.0)
            out_m = encode_full(model, masked_x, ph_vector)
            decoder_in_m = z_broadcast(out_m["graph"].node_features, out_m["z"])
            recon_m = decoder(decoder_in_m, out_m["grid_shape"])
            loss_masked = l_masked(recon_m, x, voxel_mask)

            shift_tokens, jitter_scale, jitter_offset = sample_structure_transform()
            x_g = apply_structure_transform(x, shift_tokens, jitter_scale, jitter_offset)
            h_g, _ = model.encoder.encode_hidden(x_g)
            psi_g = model.encoder.heads(h_g)
            rho_g = model.density_head(psi_g)
            slices_x, slices_g = matched_token_slices(out_x["grid_shape"], shift_tokens)
            rho_x_cropped = crop_density_state(out_x["rho"], out_x["grid_shape"], slices_x)
            rho_g_cropped = crop_density_state(rho_g, out_x["grid_shape"], slices_g)
            loss_struct = l_struct(rho_x_cropped, rho_g_cropped)

            predicted_ph = ph_predict_head(out_x["psi"], out_x["rho"])
            loss_topology = l_topology(predicted_ph, ph_vector)

            # --- NEW: L_topology_diff (formalization Section 2d). Reuses `recon` (already computed above
            # for L_recon), downsampled to match the frozen target's resolution (Section 2b), summarized
            # differentiably, and regressed against the same ph_vector via the same l_topology MSE form. ---
            t_gudhi0 = time.time()
            recon_small = F.interpolate(recon, size=(TOPOLOGY_DOWNSAMPLE,) * 3, mode="trilinear", align_corners=False)
            pred_summary_diff = differentiable_persistence_summary_batch(recon_small, max_dim=TOPOLOGY_MAX_DIM)
            loss_topology_diff = l_topology(pred_summary_diff, ph_vector)
            gudhi_s = time.time() - t_gudhi0

            total_loss = l_dirac(loss_recon, loss_masked, loss_struct, loss_topology,
                                 lam_r=LAM_R, lam_m=LAM_M, lam_s=LAM_S, lam_t=LAM_T) \
                        + LAM_TD * loss_topology_diff   # NEW term, additive on top of D_full's l_dirac verbatim

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, max_norm=MAX_GRAD_NORM)
            is_finite = bool(torch.isfinite(total_loss)) and bool(torch.isfinite(grad_norm))
            if not is_finite:
                nan_inf_steps += 1
            else:
                optimizer.step()

            topo_stability = float("nan")
            if step % TOPOLOGY_STABILITY_EVERY == 0:
                stabilities = []
                for b in range(x.shape[0]):
                    vol_g = x_g[b, 0].detach().cpu().numpy()
                    zoom = TOPOLOGY_DOWNSAMPLE / vol_g.shape[0]
                    small = ndimage.zoom(vol_g, zoom, order=1)
                    ph_g = compute_persistence_summary(small, max_dim=TOPOLOGY_MAX_DIM)
                    stabilities.append(np.linalg.norm(ph_g - ph_vector[b].cpu().numpy()))
                topo_stability = float(np.mean(stabilities))

            history["step"].append(step)
            history["loss_total"].append(float(total_loss.item()) if torch.isfinite(total_loss) else None)
            history["loss_recon"].append(float(loss_recon.item()))
            history["loss_masked"].append(float(loss_masked.item()))
            history["loss_struct"].append(float(loss_struct.item()))
            history["loss_topology"].append(float(loss_topology.item()))
            history["loss_topology_diff"].append(float(loss_topology_diff.item()))
            history["loss_topology_stability"].append(None if np.isnan(topo_stability) else topo_stability)
            history["grad_norm"].append(float(grad_norm.item()) if torch.isfinite(grad_norm) else None)
            history["gudhi_s"].append(gudhi_s)

            if step % 5 == 0:
                print("DIFFTOPO_PROGRESS " + json.dumps({
                    "epoch": epoch, "step": step, "loss_total": history["loss_total"][-1],
                    "loss_topology": history["loss_topology"][-1],
                    "loss_topology_diff": history["loss_topology_diff"][-1],
                    "grad_norm": history["grad_norm"][-1], "nan_inf_steps": nan_inf_steps,
                    "gudhi_s": round(gudhi_s, 3), "elapsed_s": round(time.time() - t0, 1),
                }), flush=True)
            step += 1

        if not SMOKE:
            save_checkpoint(model, decoder, ph_predict_head, z_broadcast, optimizer, epoch, step,
                            history, nan_inf_steps)
            log(f"checkpointed after epoch {epoch} (step {step})")

    finite_losses = [l for l in history["loss_total"] if l is not None]
    n = len(finite_losses)
    window = max(1, n // 5)
    tail = finite_losses[-window:]
    tail_first_half = tail[:len(tail) // 2] or tail
    tail_second_half = tail[len(tail) // 2:] or tail
    decreasing_in_tail = bool(np.mean(tail_second_half) <= np.mean(tail_first_half)) if n >= 2 else False
    status = "PASS" if (nan_inf_steps == 0 and decreasing_in_tail) else "FAIL"

    finite_diff_losses = [l for l in history["loss_topology_diff"] if l is not None]
    topo_diff_window = finite_diff_losses[-window:] if finite_diff_losses else []

    report = {
        "project": "dirac",
        "gate": "diff_topology_smoke",   # NEW gate id -- deliberately not "3", never overwrites gate_3.json
        "phase": "Diff-topology follow-on (dirac_diff_topology_formalization.md) - smoke run" if SMOKE
                 else "Diff-topology follow-on - full run",
        "objective": "Does adding a differentiable topology-fidelity loss on the decoder's reconstruction, "
                     "on top of D_full's unmodified recipe, converge without instability? (Section 5 real-data gate)",
        "job_id": None,
        "checks": {
            "n_steps": n,
            "n_nan_inf_steps": nan_inf_steps,
            "loss_total_first_20pct_mean": float(np.mean(finite_losses[:window])) if n else None,
            "loss_total_final_20pct_mean": float(np.mean(tail)) if n else None,
            "loss_total_final_20pct_first_half_mean": float(np.mean(tail_first_half)) if n else None,
            "loss_total_final_20pct_second_half_mean": float(np.mean(tail_second_half)) if n else None,
            "decreasing_in_tail": decreasing_in_tail,
            "grad_norm_mean": float(np.mean([g for g in history["grad_norm"] if g is not None])) if any(history["grad_norm"]) else None,
            "loss_topology_diff_first_20pct_mean": float(np.mean(finite_diff_losses[:window])) if finite_diff_losses else None,
            "loss_topology_diff_final_20pct_mean": float(np.mean(topo_diff_window)) if topo_diff_window else None,
            "gudhi_s_mean_per_step": float(np.mean(history["gudhi_s"])) if history["gudhi_s"] else None,
        },
        "status": status,
        "fallback_path": None if status == "PASS" else "reduce LAM_TD first (new term, most likely source of instability), then fall back to the existing LAM_R/LAM_M/LAM_S/LAM_T/LR sweep",
        "seeds": [SEED],
        "data_gaps": [],
        "decision_rule": "PASS iff 0 NaN/Inf steps AND mean(loss_total) over the second half of the final "
                         "20% of steps <= mean(loss_total) over the first half of that same window -- same "
                         "rule as pretrain_dirac_job.py's own Gate 3, applied to this run's total_loss "
                         "(which now includes LAM_TD * loss_topology_diff)",
        "interpretation": (
            f"{'Smoke' if SMOKE else 'Full'} run: {n} steps across {N_EPOCHS} epochs over "
            f"{len(rows)} cases, lam_td={LAM_TD}. {nan_inf_steps} NaN/Inf steps. total_loss fell from "
            f"{np.mean(finite_losses[:window]) if n else float('nan'):.4f} (first 20%) to "
            f"{np.mean(tail) if n else float('nan'):.4f} (final 20%), and within that final-20% window "
            f"{'kept decreasing' if decreasing_in_tail else 'did NOT keep decreasing'}. "
            f"loss_topology_diff fell from "
            f"{np.mean(finite_diff_losses[:window]) if finite_diff_losses else float('nan'):.4f} to "
            f"{np.mean(topo_diff_window) if topo_diff_window else float('nan'):.4f} over the same window "
            f"(PASS/FAIL is decided on total_loss alone, per decision_rule -- this number is diagnostic, "
            f"not gate-critical). Mean added gudhi wall time per step: "
            f"{np.mean(history['gudhi_s']) if history['gudhi_s'] else float('nan'):.2f}s."
        ),
        "next": (
            ("Real comparison: retrain with the SAME seed/data/steps as D_full's own gate_3 run, then rerun "
             "the H2/H3-C probe methodology against this checkpoint vs. D_full's -- separately gated, its "
             "own explicit human GO, real GPU spend" if status == "PASS"
             else "Apply the fallback (LAM_TD first) and re-run this smoke gate before proposing any full run")
        ),
        "reproduce": f"Med_Architecture_Research/dirac/jobs/train_diff_topology_job.py (HF Job). Code mounted "
                     f"read-only from hf://datasets/Chucks90/dirac-code (unmodified). Data mounted from "
                     f"hf://buckets/Chucks90/jobs-artifacts. Env: SMOKE={'1' if SMOKE else '0'} LAM_TD={LAM_TD}.",
        "human_signoff": None,
    }
    out_path = MNT / "processed" / "dirac" / ("gate_diff_topology_smoke_results.json" if SMOKE else "gate_diff_topology_full_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "history": history}, f, indent=2)
    if not SMOKE:
        save_checkpoint(model, decoder, ph_predict_head, z_broadcast, optimizer, N_EPOCHS - 1, step,
                        history, nan_inf_steps)
    else:
        ckpt_path = checkpoint_path()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(),
                   "ph_predict_head": ph_predict_head.state_dict(), "z_broadcast": z_broadcast.state_dict()},
                  ckpt_path)
    print("DIFFTOPO_RESULT " + json.dumps(report), flush=True)
    print(f"[diff-topology] HALT ({'smoke' if SMOKE else 'full'} run done) -- human review + signoff "
         f"required before {'any full run' if SMOKE else 'a probe comparison against D_full'}.", flush=True)


if __name__ == "__main__":
    main()
