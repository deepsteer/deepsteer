#!/usr/bin/env python3
"""T0.2 (STEERING_PLAN v0.2.2 §1, Amendment 1 item 5): the H1 train/held split, built once.

    python3 papers/steering/scripts/build_split.py

The 264 kept H1 scenarios are split at the scenario level into H1_train (132) and H1_held (132),
stratified by generator × register. Each stratum is shuffled with a seed derived from SEED and the
stratum key, then halved; odd strata alternate which side takes the extra scenario (in sorted
stratum order) so the totals come out 132/132. No outcome, rollout or model read enters the split.
Writes papers/steering/split_v1.json with the input sha256s, the seed and per-stratum counts.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

PAPERS = Path(__file__).resolve().parents[2]
H1_DIR = PAPERS / "kdg_panel" / "data" / "harm_divergence"
KEPT = H1_DIR / "h1_kept_ids.json"
SCENARIOS = H1_DIR / "hdiv_set.json"
OUT = PAPERS / "steering" / "split_v1.json"
SEED = 20261018
STRATA = ("generator", "register")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(ids: list[str], records: dict[str, dict], seed: int = SEED) -> dict[str, list[str]]:
    """Stratified half split; deterministic in (ids, strata, seed), independent of input order."""
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sid in sorted(ids):
        strata[tuple(str(records[sid][f]) for f in STRATA)].append(sid)
    train: list[str] = []
    held: list[str] = []
    odd_to_train = True
    for key in sorted(strata):
        members = strata[key]
        random.Random(f"{seed}|{'|'.join(key)}").shuffle(members)
        half = len(members) // 2
        if len(members) % 2:
            half += 1 if odd_to_train else 0
            odd_to_train = not odd_to_train
        train += members[:half]
        held += members[half:]
    return {"H1_train": sorted(train), "H1_held": sorted(held)}


def main() -> None:
    ids = json.loads(KEPT.read_text())["ids"]
    records = {r["id"]: r for r in json.loads(SCENARIOS.read_text())["scenarios"]}
    split = build_split(ids, records)
    counts: dict[str, dict[str, int]] = {}
    for side, members in split.items():
        for sid in members:
            key = " × ".join(str(records[sid][f]) for f in STRATA)
            counts.setdefault(key, {"H1_train": 0, "H1_held": 0})[side] += 1
    out = {
        "rule": "STEERING_PLAN v0.2.2 §1 / Amendment 1 item 5 (T0.2)",
        "strata": list(STRATA),
        "seed": SEED,
        "inputs": {
            str(KEPT.relative_to(PAPERS)): sha256(KEPT),
            str(SCENARIOS.relative_to(PAPERS)): sha256(SCENARIOS),
        },
        "n": {side: len(m) for side, m in split.items()},
        "stratum_counts": dict(sorted(counts.items())),
        **split,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(PAPERS.parent)}: {out['n']}")
    for key, c in out["stratum_counts"].items():
        print(f"  {key}: {c}")


if __name__ == "__main__":
    main()
