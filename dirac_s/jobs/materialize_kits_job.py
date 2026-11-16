# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.34", "numpy", "nibabel", "scipy"]
# ///
"""Materialize KiTS23 as 3D-volume-preserving tensors — dirac_formalization.md Section 10 /
IMPLEMENTATION_SPEC.md Phase 1. Adapted from covtoken/jobs/materialize_kits_job.py: keeps its
download/resume scaffolding and case-list logic (per-case download-then-delete, .done sentinels), but
replaces the per-slice-to-PNG loop with real 3D preprocessing that saves one preprocessed tensor per
case, per DIRAC's actual data requirement (covtoken's own pipeline discards the 3D volume; verified
directly against covtoken's real code before this was written).

Self-contained (no local package import) — matches every other job script in this vault (verified via
grep: zero sibling jobs import from a local multi-file package). `hf jobs uv run` only uploads the single
script file, so `dirac/data/*.py`'s logic is inlined here rather than imported, the same way covtoken's
and the topology/sheaf/cortexmae jobs each carry their own copy of shared helpers (e.g. split_of())
rather than sharing a local module across job files.

Segmentation masks are copied to disk for later downstream-transfer probing (spec Section 15) but their
pixel values are never read into this process — this script only ever touches image pixel data when
building the preprocessed tensor, preserving the label-free-pretraining invariant (data/leak_guard.py)
without needing the runtime guard machinery here (there's no pretraining forward pass at materialize
time to protect against).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing
from scipy import ndimage
from huggingface_hub import hf_hub_download

REPO = "neheller/KiTS-Challenge-Imaging"
MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
OUT_ROOT = MNT / "processed" / "dirac" / "kits"
MANIFEST_PATH = MNT / "processed" / "dirac" / "kits_manifest.jsonl"
TARGET_SPACING_MM = float(os.environ.get("TARGET_SPACING_MM", "1.5"))
TARGET_SHAPE = tuple(int(v) for v in os.environ.get("TARGET_SHAPE", "128,128,128").split(","))
N_CASES = int(os.environ.get("N_CASES", "489"))
MAX_CASES = int(os.environ.get("MAX_CASES", "0"))

HU_MIN, HU_MAX = -200.0, 400.0            # [CALIBRATE], per dirac_formalization.md Section 10
FOREGROUND_HU_THRESHOLD = -150.0
MIN_FOREGROUND_FRAC = 0.05


def log(m):
    print(f"[kits] {m}", flush=True)


def split_of(case_id: str) -> str:
    """Verbatim from covtoken/jobs/materialize_kits_job.py."""
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def load_canonical(path: str, target_spacing_mm: float) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(path))
    resampled = nibabel.processing.resample_to_output(img, voxel_sizes=[target_spacing_mm] * 3, order=1)
    return np.asarray(resampled.dataobj, dtype=np.float32)


def preprocess_volume(raw_hu: np.ndarray, target_shape: tuple[int, int, int]) -> dict:
    mask = raw_hu > FOREGROUND_HU_THRESHOLD
    if mask.any():
        coords = np.argwhere(mask)
        lo = np.maximum(coords.min(axis=0) - 4, 0)
        hi = np.minimum(coords.max(axis=0) + 5, raw_hu.shape)
        bbox = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    else:
        bbox = tuple(slice(0, s) for s in raw_hu.shape)
    cropped = raw_hu[bbox]
    foreground_frac = float((cropped > FOREGROUND_HU_THRESHOLD).mean())
    clipped = np.clip(cropped, HU_MIN, HU_MAX)
    normalized = ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 2.0 - 1.0).astype(np.float32)
    zoom_factors = [t / s for t, s in zip(target_shape, normalized.shape)]
    resized = ndimage.zoom(normalized, zoom_factors, order=1)
    out = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(t, r)) for t, r in zip(target_shape, resized.shape))
    out[slices] = resized[slices]
    qc = {
        "original_shape": list(raw_hu.shape), "cropped_shape": list(cropped.shape),
        "foreground_fraction": foreground_frac,
        "hu_range_seen": [float(raw_hu.min()), float(raw_hu.max())],
    }
    return {"volume": out, "qc": qc, "qc_passed": foreground_frac >= MIN_FOREGROUND_FRAC}


def process(cid: str, split: str, img_path: str, seg_path: str) -> dict | None:
    out_dir = OUT_ROOT / split / cid
    if (out_dir / ".done").exists():
        return None
    raw = load_canonical(img_path, TARGET_SPACING_MM)
    result = preprocess_volume(raw, TARGET_SHAPE)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_path = out_dir / "volume.npy"
    np.save(str(volume_path), result["volume"])
    seg_out = out_dir / "segmentation.nii.gz"
    if Path(seg_path).exists():
        # shutil.move, not Path.replace: /tmp (container-local) and /mnt (the bucket mount) are
        # different filesystems, and a plain rename() cannot cross devices (EXDEV) — move() falls
        # back to copy+delete automatically, .replace() does not and just raises.
        shutil.move(seg_path, str(seg_out))
    (out_dir / ".done").write_text("ok")
    return {"dataset": "kits23", "case_id": cid, "split": split,
           "out_path": str(volume_path.relative_to(MNT)), "qc": result["qc"], "qc_passed": result["qc_passed"]}


def main():
    t0 = time.time()
    tok = os.environ.get("HF_TOKEN")
    cases = [f"case_{i:05d}" for i in range(N_CASES)]
    if MAX_CASES:
        cases = cases[:MAX_CASES]
    log(f"cases: {len(cases)}")

    rows, errors = [], []
    done = 0
    cache = "/tmp/kits_dl"
    for i, cid in enumerate(cases):
        split = split_of(cid)
        if (OUT_ROOT / split / cid / ".done").exists():
            done += 1
            continue
        try:
            ip = hf_hub_download(REPO, f"images/{cid}.nii.gz", repo_type="dataset", token=tok, local_dir=cache)
            sp = hf_hub_download(REPO, f"segmentations/{cid}.nii.gz", repo_type="dataset", token=tok, local_dir=cache)
            row = process(cid, split, ip, sp)
            if row is not None:
                rows.append(row)
                done += 1
            try:
                os.unlink(ip)
            except OSError:
                pass
        except Exception as e:
            errors.append({"case": cid, "error": f"{type(e).__name__}: {e}"})
        if i % 5 == 0:
            print("KITS_PROGRESS " + json.dumps(
                {"processed": i + 1, "done": done, "errors": len(errors),
                 "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    res = {"dataset": "kits23", "cases": len(cases), "materialized": done, "n_errors": len(errors),
          "errors": errors[:8], "manifest": str(MANIFEST_PATH.relative_to(MNT)),
          "out_root": str(OUT_ROOT.relative_to(MNT)), "elapsed_s": round(time.time() - t0, 1)}
    print("KITS_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
