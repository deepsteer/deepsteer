#!/usr/bin/env python3
"""GPC Amendment 3 B3.5 (draft): the eval 8-gram index extended to GPC-4's outside instrument's text.

    /opt/homebrew/bin/python3.13 z_index_v2.py --impossible DIR --propensity DIR --scope DIR \
        [--agentic DIR]

v1's hashes (`cst_constitutional_steering/outputs/_sync/eval_8gram_hashes_v1.npy`) plus every word
8-gram of: ImpossibleBench's LiveCodeBench splits (the pinned dataset's parquet files: prompt, test,
original_test) and the agentic-misalignment templates (G5's continuity cell). PropensityBench is out of
GPC-4 (the author's ruling, 2026-11-17) and its text is not indexed; ScopeBench is parked and not
indexed. The --propensity and --scope flags stay for a later ruling. Only the
hashes are written (one-way, blake2b-64, `overlap_hashes`'s rule); no suite text enters the repo.
Writes ``outputs/_sync/eval_8gram_hashes_v2.npy`` and ``data/eval_8gram_v2.json`` (sha256, counts per
source, and the C_conduct and C_bounds re-checks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts")]

import overlap_hashes as OH  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/_sync/eval_8gram_hashes_v2.npy"
REC = GPC / "data/eval_8gram_v2.json"
TEXT_EXT = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".toml", ".sh", ".py",
            ".j2", ".jinja"}


def _strings(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def texts_from(d: Path, parquet_cols: tuple[str, ...] = ()) -> list[str]:
    out = []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or ".git" in f.parts or ".cache" in f.parts:
            continue
        if f.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(f)
            out += [str(v) for c in parquet_cols if c in df for v in df[c].tolist()]
        elif f.suffix == ".json":
            try:
                out += _strings(json.loads(f.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                out.append(f.read_text(errors="ignore"))
        elif f.suffix == ".jsonl":
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    out += _strings(json.loads(line))
                except json.JSONDecodeError:
                    out.append(line)
        elif f.suffix in TEXT_EXT:
            out.append(f.read_text(errors="ignore"))
    return out


def build(sources: dict[str, Path]) -> dict:
    parts, counts = [OH.load()], {"v1": int(len(OH.load()))}
    for name, d in sources.items():
        cols = ("prompt", "test", "original_test") if name == "impossiblebench_lcb" else ()
        ts = texts_from(d, cols)
        h = np.unique(np.concatenate([OH.hashes(t) for t in ts if t] or [np.zeros(0, np.uint64)]))
        counts[name] = {"texts": len(ts), "hashes": int(len(h))}
        parts.append(h)
    idx = np.unique(np.concatenate(parts))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, idx)
    rechecks = {}
    for name, f in (("c_conduct", GPC / "data/c_conduct_v1.json"),
                    ("c_bounds", GPC / "data/c_bounds_v1.json")):
        if f.exists():
            rechecks[name] = OH.shared(json.loads(f.read_text())["text"], idx)
    rec = {"file": str(OUT.relative_to(GPC)), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
           "n_hashes": int(len(idx)), "per_source": counts,
           "hash": "blake2b-64 of the space-joined word 8-gram (overlap_hashes)",
           "clause_8grams_in_v2": rechecks,
           "rule": "GPC Amendment 3 B3.5 (draft): v1 plus ImpossibleBench's LiveCodeBench splits and G5's templates"}
    REC.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("impossible", "propensity", "scope", "agentic"):
        ap.add_argument(f"--{k}", type=Path)
    a = ap.parse_args()
    src = {n: p for n, p in (("impossiblebench_lcb", a.impossible), ("propensitybench", a.propensity),
                             ("scopebench", a.scope), ("agentic_misalignment", a.agentic)) if p}
    print(json.dumps(build(src), indent=1))
