"""Local-disk materialization for bucket-mounted case data — IMPLEMENTATION_SPEC.md Phase 3, fixing the
operational finding in gate_reports/gate_2.json: per-case reads from the FUSE-mounted bucket
(hf://buckets/Chucks90/jobs-artifacts) ran ~100-150s/case vs. 0.14s/case of pure model compute, almost
certainly mount overhead rather than model cost. Not a problem when Gate 2 read each of 36 cases once; a real
problem once Phase 3 iterates the full manifest once per epoch, for many epochs. Fix: pay the bucket-read
latency once per case, in parallel, then read exclusively from local disk for every subsequent epoch.
"""
from __future__ import annotations
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _copy_one(bucket_root: Path, local_root: Path, out_path: str) -> str:
    src = bucket_root / out_path
    dst = local_root / out_path
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.rename(dst)
    return str(dst)


def materialize_local(rows: list[dict], bucket_root: Path, local_root: Path,
                      max_workers: int = 16) -> list[dict]:
    """Copies each row's out_path volume from the mounted bucket to local job disk, in parallel
    (I/O-bound — threads are fine, the GIL releases during file I/O). Idempotent: skips a case whose
    local copy already exists with a matching size, so a restarted job doesn't re-pay the copy cost.
    Returns rows with a new 'local_path' key; does not mutate the input list."""
    out_paths = [r["out_path"] for r in rows]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        local_paths = list(pool.map(lambda p: _copy_one(bucket_root, local_root, p), out_paths))
    return [{**r, "local_path": lp} for r, lp in zip(rows, local_paths)]


def _compute_one_ph(volume_path: str, downsample: int, max_dim: int) -> np.ndarray:
    # Imported inside the worker: ProcessPoolExecutor workers re-import the module fresh, and importing
    # gudhi/scipy at module load time would slow down every job that imports local_cache.py but never
    # calls this function (e.g. materialize_local-only callers).
    from scipy import ndimage
    from model.topology import compute_persistence_summary

    vol = np.load(volume_path).astype(np.float32)
    zoom = downsample / vol.shape[0]
    small = ndimage.zoom(vol, zoom, order=1)
    return compute_persistence_summary(small, max_dim=max_dim)


def materialize_ph_vectors_local(rows: list[dict], cache_root: Path, max_dim: int = 2,
                                 downsample: int = 48, max_workers: int | None = None) -> list[dict]:
    """Precomputes and caches each real case's ground-truth PH vector once, with a ProcessPoolExecutor
    (gudhi is CPU-bound — processes actually parallelize this, unlike materialize_local's I/O-bound copy).
    Same root cause as the bucket-read fix: expensive per-case work that Phase 3's repeated-epoch iteration
    would otherwise redo every epoch for no reason, since the real volume (and therefore its true topology)
    never changes. g(X)'s PH vector is NOT cached here — g is resampled every training step, so its PH
    vector has to be computed fresh each time; only the real, unperturbed X is cacheable. Rows must already
    have 'local_path' (run materialize_local first). Returns rows with a new 'local_ph_path' key."""
    max_workers = max_workers or os.cpu_count()
    cache_root.mkdir(parents=True, exist_ok=True)
    targets = []
    for r in rows:
        cache_path = cache_root / f"{r['dataset']}_{r['case_id']}_ph.npy"
        targets.append(cache_path)

    to_compute = [(r, cp) for r, cp in zip(rows, targets) if not cp.exists()]
    if to_compute:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_compute_one_ph, r["local_path"], downsample, max_dim): cp
                for r, cp in to_compute
            }
            for fut, cp in futures.items():
                np.save(cp, fut.result())

    return [{**r, "local_ph_path": str(cp)} for r, cp in zip(rows, targets)]
