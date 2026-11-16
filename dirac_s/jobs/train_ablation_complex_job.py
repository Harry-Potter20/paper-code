# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 5 ablations (IMPLEMENTATION_SPEC.md / experimental-spec Section 19): D_-complex. DIRAC-S with psi's
imaginary part never produced -- ComplexEncoder.head_im is never called (same "instantiated but untrained"
pattern used throughout the ablations: model/dirac_model.py's __init__ always builds it, this job just never
calls it), so psi = ComplexState(re=head_re(h), im=zeros). rho/graph/topology all still consume this
real-only psi through their normal, entirely unmodified complex machinery -- e.g. DensityHead's
`wim(psi.im) = wim(0) = bias_im` and `wre(psi.im) + wim(psi.re) = bias_wre + wim(psi.re)` (model/density_state.py's
basis_re/basis_im linear pairs, both carry bias by default) mean rho's basis vectors still come out genuinely
complex even though psi itself has zero phase -- a real input times a complex weight is a complex output.
This ablates psi's own phase content specifically, not the downstream complex arithmetic that consumes it,
which is the correct reading of "D_-complex" per dirac_formalization.md Section 2/9: psi is the one
representation that is complex by construction; everything built on top of it (rho, graph's feature-edge
similarity, fusion) is complex- or graph-machinery that stays fully intact.

Directly tests whether psi's phase specifically (as opposed to rho's probabilistic mixture, or graph's
relational structure, or topology's explicit supervision) contributes any of DIRAC's measured advantage --
none of H1-H5's findings implicated psi's complex structure directly (H3-C/H2 implicated L_topology; H4/H5
implicated rho's magnitude-sensitive construction), so this ablation is the least strongly pre-motivated of
the five, but is a required member of the minimum ablation set (experimental spec Section 19) and completes
the isolation of "which one of the four structural branches (complex/prob/graph/topology) actually drives
any observed gap" begun by D_-prob and D_-topology.

Uses model.encoder.head_re(h) directly rather than model.encoder.heads(h) so head_im's forward is never
even evaluated, not just discarded after the fact -- consistent with how topology_head/density_head are
"instantiated but untrained" in the sibling ablations (train_ablation_topology_job.py, and D_-graph below),
not merely zeroed downstream.

Everything else is DIRAC's full recipe, unmodified: L_recon + L_masked + L_struct + L_topology, density_head,
graph_head, and topology_head all fully active and all receive real gradient. Reuses
jobs/pretrain_dirac_job.py's L_struct/L_topology machinery verbatim (sample_structure_transform,
apply_structure_transform, matched_token_slices, crop_density_state, topology stability diagnostic) --
none of that logic depends on psi carrying a nonzero imaginary part.

No N_DIRAC-S modification -- the D_full checkpoint (pretrain_full_seed0.pt) is untouched by this job.

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

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_ablation_complex"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256   # same as D_full
STEM_STRIDE = 16
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
MAX_SHIFT_TOKENS = int(os.environ.get("MAX_SHIFT_TOKENS", "2"))
TOPOLOGY_STABILITY_EVERY = 10
SEED = int(os.environ.get("SEED", "0"))


def log(m):
    print(f"[ablation-complex] {m}", flush=True)


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


def apply_structure_transform(x: torch.Tensor, shift_tokens, jitter_scale: float, jitter_offset: float) -> torch.Tensor:
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


def encode_no_complex_phase(model: DiracModel, x: torch.Tensor, ph_vector: torch.Tensor):
    """D_-complex's forward path: identical to D_full's encode_full (jobs/pretrain_dirac_job.py) except
    psi's imaginary part is never produced -- head_im is never called (only head_re), so
    psi = ComplexState(re=head_re(h), im=zeros). rho/graph/topology are computed exactly as D_full would,
    from this real-only psi."""
    h, grid_shape = model.encoder.encode_hidden(x)
    re = model.encoder.head_re(h)
    psi = ComplexState(re=re, im=torch.zeros_like(re))
    rho = model.density_head(psi)
    graph = model.graph_head(psi)
    topology_embed = model.topology_head(ph_vector)
    z = model.fusion(rho, graph, topology_embed)
    return {"h": h, "grid_shape": grid_shape, "psi": psi, "rho": rho, "graph": graph,
           "topology_embed": topology_embed, "z": z}


def checkpoint_path() -> Path:
    return MNT / "processed" / "dirac" / "checkpoints" / f"ablation_complex_{'smoke' if SMOKE else 'full'}_seed{SEED}.pt"


def save_checkpoint(model, decoder, ph_predict_head, z_broadcast, optimizer, epoch, step, history, nan_inf_steps):
    ckpt_path = checkpoint_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(),
               "ph_predict_head": ph_predict_head.state_dict(), "z_broadcast": z_broadcast.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
               "history": history, "nan_inf_steps": nan_inf_steps}, tmp)
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
    log(f"device={device} smoke={SMOKE} epochs={N_EPOCHS} batch_size={BATCH_SIZE}")

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
              "loss_topology": [], "loss_topology_stability": [], "grad_norm": []}
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

            out_x = encode_no_complex_phase(model, x, ph_vector)
            decoder_in = z_broadcast(out_x["graph"].node_features, out_x["z"])
            recon = decoder(decoder_in, out_x["grid_shape"])
            loss_recon = l_recon(recon, x)

            token_mask = generate_block_mask(out_x["grid_shape"], mask_ratio=MASK_RATIO,
                                             batch=x.shape[0], device=device)
            voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
            masked_x = x.masked_fill(voxel_mask, 0.0)
            out_m = encode_no_complex_phase(model, masked_x, ph_vector)
            decoder_in_m = z_broadcast(out_m["graph"].node_features, out_m["z"])
            recon_m = decoder(decoder_in_m, out_m["grid_shape"])
            loss_masked = l_masked(recon_m, x, voxel_mask)

            shift_tokens, jitter_scale, jitter_offset = sample_structure_transform()
            x_g = apply_structure_transform(x, shift_tokens, jitter_scale, jitter_offset)
            h_g, _ = model.encoder.encode_hidden(x_g)
            re_g = model.encoder.head_re(h_g)
            psi_g = ComplexState(re=re_g, im=torch.zeros_like(re_g))
            rho_g = model.density_head(psi_g)
            slices_x, slices_g = matched_token_slices(out_x["grid_shape"], shift_tokens)
            rho_x_cropped = crop_density_state(out_x["rho"], out_x["grid_shape"], slices_x)
            rho_g_cropped = crop_density_state(rho_g, out_x["grid_shape"], slices_g)
            loss_struct = l_struct(rho_x_cropped, rho_g_cropped)

            predicted_ph = ph_predict_head(out_x["psi"], out_x["rho"])
            loss_topology = l_topology(predicted_ph, ph_vector)

            total_loss = l_dirac(loss_recon, loss_masked, loss_struct, loss_topology,
                                 lam_r=LAM_R, lam_m=LAM_M, lam_s=LAM_S, lam_t=LAM_T)

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
            history["loss_topology_stability"].append(None if np.isnan(topo_stability) else topo_stability)
            history["grad_norm"].append(float(grad_norm.item()) if torch.isfinite(grad_norm) else None)

            if step % 5 == 0:
                print("ABLATION_COMPLEX_PROGRESS " + json.dumps({
                    "epoch": epoch, "step": step, "loss_total": history["loss_total"][-1],
                    "grad_norm": history["grad_norm"][-1], "nan_inf_steps": nan_inf_steps,
                    "elapsed_s": round(time.time() - t0, 1),
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

    report = {
        "ablation": "D_-complex", "n_params_dirac_full_locked": 1552520,
        "mode": "smoke" if SMOKE else "full",
        "checks": {
            "n_steps": n, "n_nan_inf_steps": nan_inf_steps,
            "loss_total_first_20pct_mean": float(np.mean(finite_losses[:window])) if n else None,
            "loss_total_final_20pct_mean": float(np.mean(tail)) if n else None,
            "decreasing_in_tail": decreasing_in_tail,
            "grad_norm_mean": float(np.mean([g for g in history["grad_norm"] if g is not None])) if any(history["grad_norm"]) else None,
        },
        "status": "PASS" if (nan_inf_steps == 0 and decreasing_in_tail) else "FAIL",
        "seeds": [SEED],
        "reproduce": f"Med_Architecture_Research/dirac/jobs/train_ablation_complex_job.py (HF Job). Env: SMOKE={'1' if SMOKE else '0'}.",
    }
    out_path = MNT / "processed" / "dirac" / ("ablation_complex_smoke_results.json" if SMOKE else "ablation_complex_full_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "history": history}, f, indent=2)
    save_checkpoint(model, decoder, ph_predict_head, z_broadcast, optimizer, N_EPOCHS - 1, step, history, nan_inf_steps)
    print("ABLATION_COMPLEX_RESULT " + json.dumps(report), flush=True)
    print(f"[ablation-complex] HALT ({'smoke' if SMOKE else 'full'} run done) — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
