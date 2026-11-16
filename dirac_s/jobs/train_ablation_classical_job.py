# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "gudhi"]
# ///
# gudhi is a transitive import (model.dirac_model -> model.topology imports it at module level for
# compute_persistence_summary, even though this job never calls that function) -- see
# jobs/probe_linear_job.py's header comment for the full explanation of this exact class of bug.
"""Phase 5 ablations (IMPLEMENTATION_SPEC.md / experimental-spec Section 19): D_classical. DIRAC's own
conv/attention stem, with the complex/probabilistic/graph/topology branches stripped, trained with DIRAC's
own pretraining recipe -- dirac_formalization.md Section 9: "answers what do the DIRAC-specific components
add over the shared stem, holding the recipe fixed." NOT the same thing as the parameter-matched CNN/ViT
baselines (Phase 4) -- those are independent architectures with their own standard recipe; this is DIRAC's
own stem with nothing removed from *how* it trains, just *what* it has access to.

Already fully built: DiracModel(classical_ablation=True) + classical_pool exist since Phase 0b
(model/dirac_model.py). With classical_ablation=True, forward() returns rho=graph=topology=None -- so
L_struct/L_topology are not just zeroed, they are structurally uncomputable (no rho to feed them). Recipe is
therefore L_recon+L_masked only, mirroring the CNN/ViT baselines' own recipe almost exactly -- the honest
reading of "holding the recipe fixed" once two of its four terms can't exist without the stripped branches.

Reconstruction path: z_broadcast(h, z), reusing ZBroadcast (model/losses.py) exactly as built for D_full,
just fed h (the shared per-token stem output) instead of graph.node_features -- decoding straight from h
alone would leave classical_pool (and therefore z, the thing this ablation is actually about) at its random
init all through training, since z would never appear in any loss. Broadcasting z onto h fixes that: the
decoder gets both the per-token stem output and z's own trained contribution to every token, same mechanism
D_full already uses for exactly this reason.

Called from jobs/probe_h3_topology_job.py's style rather than jobs/pretrain_dirac_job.py's -- this ablation
skips graph_head/density_head/topology_head entirely by never calling them (not by zeroing their output), so
its own encode step is simpler than a straight copy of pretrain_dirac_job.py's encode_full.

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

from model.dirac_model import DiracModel
from model.losses import Decoder3D, ZBroadcast, generate_block_mask, upsample_token_mask_to_voxels, l_recon, l_masked, l_dirac
from data.local_cache import materialize_local

MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
LOCAL = Path(os.environ.get("DIRAC_LOCAL_ROOT", "/local_cache_ablation_classical"))
D, K, GRID_SHAPE, Z_DIM = 64, 8, (8, 8, 8), 256   # same as D_full -- classical_ablation doesn't change these
STEM_STRIDE = 16
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


def log(m):
    print(f"[ablation-classical] {m}", flush=True)


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
    vols = np.stack([np.load(r["local_path"]).astype(np.float32) for r in rows])
    return torch.from_numpy(vols).unsqueeze(1).to(device)


def checkpoint_path() -> Path:
    return MNT / "processed" / "dirac" / "checkpoints" / f"ablation_classical_{'smoke' if SMOKE else 'full'}_seed{SEED}.pt"


def save_checkpoint(model, decoder, z_broadcast, optimizer, epoch, step, history, nan_inf_steps):
    ckpt_path = checkpoint_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(),
               "z_broadcast": z_broadcast.state_dict(), "optimizer": optimizer.state_dict(),
               "epoch": epoch, "step": step, "history": history, "nan_inf_steps": nan_inf_steps}, tmp)
    tmp.rename(ckpt_path)


def load_checkpoint_if_present(model, decoder, z_broadcast, optimizer, device):
    ckpt_path = checkpoint_path()
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    z_broadcast.load_state_dict(ckpt["z_broadcast"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["step"], ckpt["history"], ckpt["nan_inf_steps"]


def encode_classical(model: DiracModel, x: torch.Tensor):
    """The D_classical forward path: shared stem only, no complex/prob/graph/topology branches at all."""
    h, grid_shape = model.encoder.encode_hidden(x)
    z = model.classical_pool(h.mean(dim=1))
    return h, z, grid_shape


def forward_recon(model, decoder, z_broadcast, x: torch.Tensor):
    h, z, grid_shape = encode_classical(model, x)
    decoder_in = z_broadcast(h, z)
    return decoder(decoder_in, grid_shape), grid_shape


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} smoke={SMOKE} epochs={N_EPOCHS} batch_size={BATCH_SIZE}")

    rows = load_train_rows()
    log(f"materializing {len(rows)} train cases to local disk")
    rows = materialize_local(rows, bucket_root=MNT, local_root=LOCAL)

    model = DiracModel(d=D, K=K, grid_shape=GRID_SHAPE, z_dim=Z_DIM).to(device)
    decoder = Decoder3D(in_channels=D, widths=(D, D // 2, D // 4, 1)).to(device)
    z_broadcast = ZBroadcast(z_dim=Z_DIM, d=D).to(device)
    params = list(model.parameters()) + list(decoder.parameters()) + list(z_broadcast.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    history = {"step": [], "loss_total": [], "loss_recon": [], "loss_masked": [], "grad_norm": []}
    nan_inf_steps = 0
    step = 0
    start_epoch = 0
    if not SMOKE and os.environ.get("RESUME", "1") == "1":
        resumed = load_checkpoint_if_present(model, decoder, z_broadcast, optimizer, device)
        if resumed is not None:
            start_epoch, step, history, nan_inf_steps = resumed
            start_epoch += 1
            log(f"resumed from checkpoint: epoch={start_epoch} step={step}")
    t0 = time.time()

    for epoch in range(start_epoch, N_EPOCHS):
        random.shuffle(rows)
        for i in range(0, len(rows) - len(rows) % BATCH_SIZE, BATCH_SIZE):
            batch_rows = rows[i:i + BATCH_SIZE]
            x = load_batch(batch_rows, device)

            recon, grid_shape = forward_recon(model, decoder, z_broadcast, x)
            loss_recon = l_recon(recon, x)

            token_mask = generate_block_mask(grid_shape, mask_ratio=MASK_RATIO, batch=x.shape[0], device=device)
            voxel_mask = upsample_token_mask_to_voxels(token_mask, stem_stride=STEM_STRIDE)
            masked_x = x.masked_fill(voxel_mask, 0.0)
            recon_m, _ = forward_recon(model, decoder, z_broadcast, masked_x)
            loss_masked = l_masked(recon_m, x, voxel_mask)

            zero = loss_recon.new_zeros(())
            total_loss = l_dirac(loss_recon, loss_masked, zero, zero, lam_r=LAM_R, lam_m=LAM_M, lam_s=0.0, lam_t=0.0)

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
            history["grad_norm"].append(float(grad_norm.item()) if torch.isfinite(grad_norm) else None)

            if step % 5 == 0:
                print("ABLATION_CLASSICAL_PROGRESS " + json.dumps({
                    "epoch": epoch, "step": step, "loss_total": history["loss_total"][-1],
                    "grad_norm": history["grad_norm"][-1], "nan_inf_steps": nan_inf_steps,
                    "elapsed_s": round(time.time() - t0, 1),
                }), flush=True)
            step += 1

        if not SMOKE:
            save_checkpoint(model, decoder, z_broadcast, optimizer, epoch, step, history, nan_inf_steps)
            log(f"checkpointed after epoch {epoch} (step {step})")

    finite_losses = [l for l in history["loss_total"] if l is not None]
    n = len(finite_losses)
    window = max(1, n // 5)
    tail = finite_losses[-window:]
    tail_first_half = tail[:len(tail) // 2] or tail
    tail_second_half = tail[len(tail) // 2:] or tail
    decreasing_in_tail = bool(np.mean(tail_second_half) <= np.mean(tail_first_half)) if n >= 2 else False

    report = {
        "ablation": "D_classical", "n_params_dirac_full_locked": 1552520,
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
        "reproduce": f"Med_Architecture_Research/dirac/jobs/train_ablation_classical_job.py (HF Job). Env: SMOKE={'1' if SMOKE else '0'}.",
    }
    out_path = MNT / "processed" / "dirac" / ("ablation_classical_smoke_results.json" if SMOKE else "ablation_classical_full_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "history": history}, f, indent=2)
    save_checkpoint(model, decoder, z_broadcast, optimizer, N_EPOCHS - 1, step, history, nan_inf_steps)
    print("ABLATION_CLASSICAL_RESULT " + json.dumps(report), flush=True)
    print(f"[ablation-classical] HALT ({'smoke' if SMOKE else 'full'} run done) — results at {out_path}", flush=True)


if __name__ == "__main__":
    main()
