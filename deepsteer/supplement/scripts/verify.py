#!/usr/bin/env python3
"""Verify the supplement's integrity and the shared-arrays-live-once invariant.

Two checks, both zero-dependency and read-only:

1. Integrity: recompute every artifact's sha256 and compare to MANIFEST.json.
   Fails if a distilled artifact drifted from its indexed hash.

2. Shared-arrays-live-once: the canonical shared CSVs
   (bottleneck_pr, depth_asymmetry) have per-paper plotting mirrors under each
   paper's figure_data/. This asserts the mirrors carry the same values as the
   canonical file on the columns they share, so a number can only be changed in
   one place. Reports drift instead of silently tolerating two sources of truth.

    python3 deepsteer/supplement/scripts/verify.py

Exit non-zero on any mismatch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(ROOT, "..", ".."))

# Canonical shared CSV -> the per-paper plotting mirrors that must agree with it
# on the key column. (model is the join key for both.)
MIRRORS = {
    "figure_data/bottleneck_pr.csv": {
        "key": ["model"],
        "check_cols": {"decision_site_pr", "content_pr"},
        "mirrors": [
            "papers/figure_data/mn_bottleneck_pr.csv",
            "papers/fl_what_refusal_reads/figure_data/fl_bottleneck_pr.csv",
        ],
    },
    "figure_data/depth_asymmetry.csv": {
        "key": ["model", "layer"],
        "check_cols": {"A", "ci_low", "ci_high"},
        "mirrors": [
            "papers/figure_data/mn_depth_collapse.csv",
            "papers/fl_what_refusal_reads/figure_data/fl_depth_collapse.csv",
        ],
    },
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_integrity() -> list[str]:
    errs = []
    manifest = json.load(open(os.path.join(ROOT, "MANIFEST.json")))
    for a in manifest["artifacts"]:
        p = os.path.join(ROOT, a["path"])
        if not os.path.exists(p):
            errs.append(f"integrity: missing {a['path']}")
            continue
        got = sha256(p)
        if got != a["sha256"]:
            errs.append(f"integrity: sha mismatch {a['path']} (manifest {a['sha256'][:12]}, disk {got[:12]})")
    return errs


def check_mirrors() -> list[str]:
    errs = []
    def keyof(row: dict, cols: list[str]) -> tuple:
        return tuple((row.get(c) or "").strip() for c in cols)

    for canon_rel, spec in MIRRORS.items():
        cols = spec["key"]
        canon = {keyof(r, cols): r for r in load_rows(os.path.join(ROOT, canon_rel))}
        for m_rel in spec["mirrors"]:
            m_path = os.path.join(REPO, m_rel)
            if not os.path.exists(m_path):
                errs.append(f"mirror: missing {m_rel}")
                continue
            for r in load_rows(m_path):
                k = keyof(r, cols)
                base = canon.get(k)
                if base is None:
                    errs.append(f"mirror: {m_rel} has key {k} absent from canonical {canon_rel}")
                    continue
                for col in spec["check_cols"] & set(r) & set(base):
                    if (r[col] or "").strip() != (base[col] or "").strip():
                        errs.append(
                            f"mirror drift: {m_rel} [{k}].{col}={r[col]!r} != canonical {base[col]!r}"
                        )
    return errs


def main() -> int:
    errs = check_integrity() + check_mirrors()
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print("OK: manifest hashes match; shared-array mirrors agree with canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
