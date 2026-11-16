# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "gudhi"]
# ///
"""Phase 4 (IMPLEMENTATION_SPEC.md): trains the parameter-matched CNN3D baseline (baselines/cnn3d.py,
width=36, calibrated in Phase 0b to configs/param_ledger.json's N_DIRAC-S within 0.405%). Standard MAE-style
reconstruction only (L_recon + L_masked) -- not DIRAC's L_struct/L_topology, which have no non-contrived CNN
equivalent (formalization Section 9: the parameter-matched baselines get "their own standard training
recipe," not DIRAC's own recipe held fixed -- that's what the separate D_classical ablation is for).

Same optimization budget as DIRAC-S's own full run (jobs/pretrain_dirac_job.py, gate_reports/gate_3.json):
same 533 real train cases, 50 epochs, batch size 2, AdamW lr=1e-4, grad-clip 1.0, mask ratio 0.75 -- isolates
the comparison to architecture + recipe, per experimental-spec Section 17's "comparable optimization budget."

CNN3DBaseline.encode_hidden() (added this phase) exposes the pre-pool (B,288,8,8,8) spatial feature map --
forward()'s AdaptiveAvgPool3d discards spatial structure entirely, which a single pooled vector can't be
decoded back into a volume from. Decoder3D and the masking utilities are reused unmodified from
model/losses.py: CNN3DBaseline downsamples 128^3 -> 8^3 via four stride-2 blocks, the same 16x total factor
DIRAC's own stem uses, so stem_stride=16 and a 4-stage Decoder3D apply as-is.

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

from baselines.cnn3d import CNN3DBaseline
from model.losses import Decoder3D, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_dirac, l_topology
from model.param_count import exact_param_count
from data.local_cache import materialize_local, materialize_ph_vectors_local
from model.topology import TopologyHead

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_cnn"))
WIDTH, DEPTH, Z_DIM = 36, 4, 256       # locked in configs/param_ledger.json (Phase 0b) — do not change
CNN_OUT_CHANNELS = WIDTH * (2 ** (DEPTH - 1))   # 288 at width=36, depth=4
STEM_STRIDE = 16                        # 4 stride-2 blocks = 16x total downsample, matches DIRAC's stem
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
SEED = int(os.environ.get("SEED", "0"))
# R1 (RECIPES_R1_R2_PREREG_2026-11-10.md): AUX_TOPO=1 adds DIRAC's own L_topology as an auxiliary head on the
# mean-pooled encoder features -- same cached gudhi target, same MSE, same LAM_T=1.0 default as pretrain_dirac_job.py.
AUX_TOPO = os.environ.get("AUX_TOPO", "0") == "1"
LAM_T = float(os.environ.get("LAM_T", "1.0"))
TOPOLOGY_MAX_DIM, TOPOLOGY_DOWNSAMPLE = 2, 48


def log(m):
    print(f"[baseline-cnn] {m}", flush=True)


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


def load_batch(rows: list[dict], device) -> torch.Tensor:
    vols = np.stack([np.load(r["local_path"]).astype(np.float32) for r in rows])   # (B,128,128,128)
    x = torch.from_numpy(vols).unsqueeze(1).to(device)      # (B,1,128,128,128)
    if not AUX_TOPO:
        return x, None
    phs = np.stack([np.load(r["local_ph_path"]).astype(np.float32) for r in rows])   # (B,m)
    return x, torch.from_numpy(phs).to(device)


def checkpoint_path() -> Path:
    return MNT / "processed" / "dirac" / "checkpoints" / f"baseline_cnn{'_topo' if AUX_TOPO else ''}_{'smoke' if SMOKE else 'full'}_seed{SEED}.pt"


def save_checkpoint(model, decoder, optimizer, epoch, step, history, nan_inf_steps, topo_head=None):
    ckpt_path = checkpoint_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
               "epoch": epoch, "step": step, "history": history, "nan_inf_steps": nan_inf_steps,
               "topo_head": topo_head.state_dict() if topo_head is not None else None}, tmp)
    tmp.rename(ckpt_path)


def load_checkpoint_if_present(model, decoder, optimizer, device, topo_head=None):
    ckpt_path = checkpoint_path()
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if topo_head is not None and ckpt.get("topo_head"):
        topo_head.load_state_dict(ckpt["topo_head"])
    return ckpt["epoch"], ckpt["step"], ckpt["history"], ckpt["nan_inf_steps"]


def forward_recon(model: CNN3DBaseline, decoder: Decoder3D, x: torch.Tensor, grid_shape):
    feat = model.encode_hidden(x)                              # (B, C, dz, hy, wx)
    tokens = feat.flatten(2).transpose(1, 2)                     # (B, N, C)
    return decoder(tokens, grid_shape), feat.mean(dim=(2, 3, 4))


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} smoke={SMOKE} epochs={N_EPOCHS} batch_size={BATCH_SIZE}")

    rows = load_train_rows()
    log(f"materializing {len(rows)} train cases to local disk")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)

    model = CNN3DBaseline(width=WIDTH, depth=DEPTH, z_dim=Z_DIM).to(device)
    decoder = Decoder3D(in_channels=CNN_OUT_CHANNELS, widths=(128, 64, 32, 1)).to(device)
    topo_head = None
    if AUX_TOPO:
        rows = materialize_ph_vectors_local(rows, cache_root=LOCAL / "ph_cache", max_dim=TOPOLOGY_MAX_DIM,
                                            downsample=TOPOLOGY_DOWNSAMPLE)
        m = TopologyHead.summary_length(TOPOLOGY_MAX_DIM)
        topo_head = nn.Sequential(nn.Linear(CNN_OUT_CHANNELS, 64), nn.GELU(), nn.Linear(64, m)).to(device)
        log(f"AUX_TOPO on: head {CNN_OUT_CHANNELS}->64->{m}, lam_t={LAM_T}")
    params = list(model.parameters()) + list(decoder.parameters()) + (list(topo_head.parameters()) if topo_head is not None else [])
    optimizer = torch.optim.AdamW(params, lr=LR)

    history = {"step": [], "loss_total": [], "loss_recon": [], "loss_masked": [], "loss_topology": [], "grad_norm": []}
    nan_inf_steps = 0
    step = 0
    start_epoch = 0
    if not SMOKE and os.environ.get("RESUME", "1") == "1":
        resumed = load_checkpoint_if_present(model, decoder, optimizer, device, topo_head)
        if resumed is not None:
            start_epoch, step, history, nan_inf_steps = resumed
            start_epoch += 1
            log(f"resumed from checkpoint: epoch={start_epoch} step={step}")
    t0 = time.time()
    grid_shape = None

    for epoch in range(start_epoch, N_EPOCHS):
        random.shuffle(rows)
        for i in range(0, len(rows) - len(rows) % BATCH_SIZE, BATCH_SIZE):
            batch_rows = rows[i:i + BATCH_SIZE]
            x, ph_vector = load_batch(batch_rows, device)
            if grid_shape is None:
                d = x.shape[2] // STEM_STRIDE
                grid_shape = (d, d, d)

            recon, pooled = forward_recon(model, decoder, x, grid_shape)
            loss_recon = l_recon(recon, x)

            token_mask = generate_block_mask(grid_shape, mask_ratio=MASK_RATIO, batch=x.shape[0], device=device)
            voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
            masked_x = x.masked_fill(voxel_mask, 0.0)
            recon_m, _ = forward_recon(model, decoder, masked_x, grid_shape)
            loss_masked = l_masked(recon_m, x, voxel_mask)

            zero = loss_recon.new_zeros(())
            loss_topology = l_topology(topo_head(pooled), ph_vector) if AUX_TOPO else zero
            total_loss = l_dirac(loss_recon, loss_masked, zero, loss_topology, lam_r=LAM_R, lam_m=LAM_M, lam_s=0.0,
                                 lam_t=LAM_T if AUX_TOPO else 0.0)

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, max_norm=MAX_GRAD_NORM)
            is_finite = bool(torch.isfinite(total_loss)) and bool(torch.isfinite(grad_norm))
            if not is_finite:
                nan_inf_steps += 1
            else:
                optimizer.step()

            history["step"].append(step)
            history["loss_total"].append(float(total_loss.item()) if torch.isfinite(total_loss) else None)
            history["loss_recon"].append(float(loss_recon.item()))
            history["loss_masked"].append(float(loss_masked.item()))
            history["loss_topology"].append(float(loss_topology.item()))
            history["grad_norm"].append(float(grad_norm.item()) if torch.isfinite(grad_norm) else None)

            if step % 5 == 0:
                print("BASELINE_CNN_PROGRESS " + json.dumps({
                    "epoch": epoch, "step": step, "loss_total": history["loss_total"][-1],
                    "grad_norm": history["grad_norm"][-1], "nan_inf_steps": nan_inf_steps,
                    "elapsed_s": round(time.time() - t0, 1),
                }), flush=True)
            step += 1

        if not SMOKE:
            save_checkpoint(model, decoder, optimizer, epoch, step, history, nan_inf_steps, topo_head)
            log(f"checkpointed after epoch {epoch} (step {step})")

    finite_losses = [l for l in history["loss_total"] if l is not None]
    n = len(finite_losses)
    window = max(1, n // 5)
    tail = finite_losses[-window:]
    tail_first_half = tail[:len(tail) // 2] or tail
    tail_second_half = tail[len(tail) // 2:] or tail
    decreasing_in_tail = bool(np.mean(tail_second_half) <= np.mean(tail_first_half)) if n >= 2 else False
    n_params = exact_param_count(model)

    report = {
        "architecture": "cnn3d", "aux_topo": AUX_TOPO, "lam_t": LAM_T if AUX_TOPO else 0.0, "width": WIDTH, "n_params": n_params,
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
        "reproduce": f"Med_Architecture_Research/dirac/jobs/train_baseline_cnn_job.py (HF Job). Env: SMOKE={'1' if SMOKE else '0'}.",
    }
    out_path = MNT / "processed" / "dirac" / (f"baseline_cnn{'_topo' if AUX_TOPO else ''}_{'smoke' if SMOKE else 'full'}_seed{SEED}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "history": history}, f, indent=2)
    save_checkpoint(model, decoder, optimizer, N_EPOCHS - 1, step, history, nan_inf_steps, topo_head)
    print("BASELINE_CNN_RESULT " + json.dumps(report), flush=True)
    print(f"[baseline-cnn] HALT ({'smoke' if SMOKE else 'full'} run done) — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
