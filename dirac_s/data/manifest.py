"""Patient-level split hashing (reused verbatim from covtoken/jobs/materialize_kits_job.py's split_of)
and the manifest schema — spec Section 10 ("all splits must be patient-level"). Freezing writes
configs/dataset_manifest.lock.json, per IMPLEMENTATION_SPEC.md Phase 1.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def split_of(case_id: str) -> str:
    """Verbatim from covtoken: sha256(case_id) % 100 -> 70/15/15 train/val/test. Deterministic,
    reproducible, no leakage by construction — reused exactly, not reimplemented."""
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def manifest_row(dataset: str, case_id: str, out_path: str, qc: dict, qc_passed: bool) -> dict:
    return {
        "dataset": dataset,
        "case_id": case_id,
        "split": split_of(case_id),
        "out_path": out_path,
        "qc": qc,
        "qc_passed": qc_passed,
    }


def write_manifest(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_no_split_leakage(rows: list[dict]) -> list[str]:
    """Every case_id must map to exactly one split, consistently — a real leakage test, not just
    trusting split_of()'s determinism. Returns the list of leaking case_ids (empty = PASS)."""
    seen: dict[str, str] = {}
    leaks = []
    for row in rows:
        cid, split = row["case_id"], row["split"]
        if cid in seen and seen[cid] != split:
            leaks.append(cid)
        seen[cid] = split
    return leaks


def freeze_manifest(rows: list[dict], datasets_included: list[str], lock_path: str) -> dict:
    """Gate 1's terminal artifact: configs/dataset_manifest.lock.json. Not modified afterward except
    by adding a new, separately-gated dataset (IMPLEMENTATION_SPEC.md Section 7)."""
    passed = [r for r in rows if r["qc_passed"]]
    frozen = {
        "datasets_included": datasets_included,
        "n_cases_total": len(rows),
        "n_cases_qc_passed": len(passed),
        "qc_pass_rate": round(len(passed) / len(rows), 4) if rows else 0.0,
        "split_counts": {
            split: sum(1 for r in passed if r["split"] == split)
            for split in ("train", "val", "test")
        },
        "leakage_check": check_no_split_leakage(rows),
    }
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        json.dump(frozen, f, indent=2)
    return frozen
