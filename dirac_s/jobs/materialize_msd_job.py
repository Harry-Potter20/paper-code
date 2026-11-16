# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "nibabel", "scipy", "requests"]
# ///
"""Materialize an MSD task (Liver Task03 or Pancreas Task07) as 3D-volume-preserving tensors — spec
Section 10. Adapted from covtoken/jobs/materialize_msd_pancreas_job.py: keeps its tar-download,
random-access-extract, and MSD_TASK/OUT_SUB parameterization (covtoken already reuses this one script
for multiple MSD tasks via env vars — reused as-is rather than forked into two near-duplicate files),
but replaces the per-slice-to-PNG loop with real 3D preprocessing.

Self-contained (no local package import) — see materialize_kits_job.py's docstring for why; same fix
applied here after the identical ModuleNotFoundError on the first real HF Jobs run.

Run with MSD_TASK=Task03_Liver OUT_SUB=liver for liver, or the Task07_Pancreas defaults below for
pancreas.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing
from scipy import ndimage
import requests

TASK = os.environ.get("MSD_TASK", "Task07_Pancreas")
OUT_SUB = os.environ.get("OUT_SUB", "pancreas")
TAR_URL = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{TASK}.tar"
MNT = Path(os.environ.get("DIRAC_OUT_ROOT", "/mnt"))
TAR_LOCAL = Path(f"/tmp/{TASK}.tar")
OUT_ROOT = MNT / "processed" / "dirac" / OUT_SUB
MANIFEST_PATH = MNT / "processed" / "dirac" / f"{OUT_SUB}_manifest.jsonl"
TARGET_SPACING_MM = float(os.environ.get("TARGET_SPACING_MM", "1.5"))
TARGET_SHAPE = tuple(int(v) for v in os.environ.get("TARGET_SHAPE", "128,128,128").split(","))
MAX_CASES = int(os.environ.get("MAX_CASES", "0"))
MIN_TAR_BYTES = int(os.environ.get("MIN_TAR_BYTES", "1000000"))

HU_MIN, HU_MAX = -200.0, 400.0
FOREGROUND_HU_THRESHOLD = -150.0
MIN_FOREGROUND_FRAC = 0.05


def log(m):
    print(f"[msd:{OUT_SUB}] {m}", flush=True)


def split_of(case_id: str) -> str:
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def download():
    if TAR_LOCAL.exists() and TAR_LOCAL.stat().st_size > MIN_TAR_BYTES:
        log("tar already present")
        return
    log(f"downloading {TAR_URL} ...")
    with requests.get(TAR_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(TAR_LOCAL, "wb") as f:
            t0, got = time.time(), 0
            for chunk in r.iter_content(chunk_size=1 << 23):
                f.write(chunk)
                got += len(chunk)
                if got % (1 << 30) < (1 << 23):
                    log(f"  downloaded {got/1e9:.1f}GB elapsed={time.time()-t0:.0f}s")
    log("download complete")


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


def process_case(tf: tarfile.TarFile, img_member, lbl_member, case_id: str, split: str) -> dict | None:
    out_dir = OUT_ROOT / split / case_id
    if (out_dir / ".done").exists():
        return None
    tmp = Path("/tmp/msd_case")
    tmp.mkdir(exist_ok=True)
    for m in (img_member, lbl_member):
        tf.extract(m, path=str(tmp))
    img_path = str(tmp / img_member.name)
    seg_path = str(tmp / lbl_member.name)

    raw = load_canonical(img_path, TARGET_SPACING_MM)
    result = preprocess_volume(raw, TARGET_SHAPE)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_path = out_dir / "volume.npy"
    np.save(str(volume_path), result["volume"])
    seg_out = out_dir / "segmentation.nii.gz"
    # shutil.move, not Path.replace — see materialize_kits_job.py's identical fix (EXDEV across
    # /tmp container-local storage and the /mnt bucket mount).
    shutil.move(seg_path, str(seg_out))
    (out_dir / ".done").write_text("ok")
    try:
        (tmp / img_member.name).unlink()
    except OSError:
        pass
    return {"dataset": OUT_SUB, "case_id": case_id, "split": split,
           "out_path": str(volume_path.relative_to(MNT)), "qc": result["qc"], "qc_passed": result["qc_passed"]}


def main():
    t0 = time.time()
    download()
    tf = tarfile.open(str(TAR_LOCAL), "r:")
    members = tf.getmembers()
    imgs = {Path(m.name).name: m for m in members
           if "/imagesTr/" in m.name and m.name.endswith(".nii.gz") and not Path(m.name).name.startswith(".")}
    lbls = {Path(m.name).name: m for m in members
           if "/labelsTr/" in m.name and m.name.endswith(".nii.gz") and not Path(m.name).name.startswith(".")}
    cases = sorted(set(imgs) & set(lbls))
    if MAX_CASES:
        cases = cases[:MAX_CASES]
    log(f"cases: {len(cases)}")

    rows, errors = [], []
    done = 0
    for i, name in enumerate(cases):
        cid = name.replace(".nii.gz", "")
        split = split_of(cid)
        try:
            row = process_case(tf, imgs[name], lbls[name], cid, split)
            if row is not None:
                rows.append(row)
            done += 1
        except Exception as e:
            errors.append({"case": cid, "error": f"{type(e).__name__}: {e}"})
        if i % 5 == 0:
            print(f"MSD_{OUT_SUB.upper()}_PROGRESS " + json.dumps(
                {"processed": i + 1, "done": done, "errors": len(errors),
                 "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    res = {"dataset": OUT_SUB, "task": TASK, "cases": len(cases), "materialized": done,
          "n_errors": len(errors), "errors": errors[:8], "manifest": str(MANIFEST_PATH.relative_to(MNT)),
          "out_root": str(OUT_ROOT.relative_to(MNT)), "elapsed_s": round(time.time() - t0, 1)}
    print(f"MSD_{OUT_SUB.upper()}_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
