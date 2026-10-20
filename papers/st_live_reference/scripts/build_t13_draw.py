#!/usr/bin/env python3
"""T1.3's scenario draw (ST_PLAN §5.0; author, 2026-10-19), built once from T1.1's engaged set.

    python3 papers/st_live_reference/scripts/build_t13_draw.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b/select/engaged.json

Gate: 20 of the brake-engaged H1_train scenarios, stratified by the split's strata (generator ×
register). Each stratum's quota is its share of the engaged set times 20, rounded by largest
remainder (ties to the earlier stratum in sorted order); each stratum is shuffled with a seed
derived from SEED and the stratum key and its quota taken from the front. Sweep: 5 of the
remaining engaged scenarios, shuffled with a seed derived from SEED, so the strength sweep never
tunes alpha on a gate scenario. No outcome beyond the engaged rule enters the draw. Writes
papers/st_live_reference/data/t13_draw_v1.json with the input sha256s, the seed and the quotas.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_split as BS  # noqa: E402

OUT = HERE.parent / "data" / "t13_draw_v1.json"
SEED = 0
N_GATE, N_SWEEP = 20, 5


def quotas(counts: dict[tuple[str, ...], int], n: int) -> dict[tuple[str, ...], int]:
    """Largest-remainder allocation of ``n`` across strata in proportion to ``counts``."""
    total = sum(counts.values())
    exact = {k: n * c / total for k, c in counts.items()}
    q = {k: int(v) for k, v in exact.items()}
    order = sorted(counts, key=lambda k: (-(exact[k] - q[k]), k))
    for k in order[: n - sum(q.values())]:
        q[k] += 1
    return q


def draw(engaged: list[str], records: dict[str, dict], seed: int = SEED) -> dict:
    """Stratified gate draw and the disjoint sweep draw; deterministic in (ids, strata, seed)."""
    strata: dict[tuple[str, ...], list[str]] = {}
    for sid in sorted(engaged):
        strata.setdefault(tuple(str(records[sid][f]) for f in BS.STRATA), []).append(sid)
    q = quotas({k: len(v) for k, v in strata.items()}, N_GATE)
    gate: list[str] = []
    for key in sorted(strata):
        members = list(strata[key])
        random.Random(f"{seed}|{'|'.join(key)}").shuffle(members)
        gate += members[: q[key]]
    rest = sorted(set(engaged) - set(gate))
    random.Random(f"{seed}|sweep").shuffle(rest)
    return {"gate": sorted(gate), "sweep": sorted(rest[:N_SWEEP]),
            "quotas": {" × ".join(k): q[k] for k in sorted(q)},
            "engaged_per_stratum": {" × ".join(k): len(v) for k, v in sorted(strata.items())}}


def main() -> None:
    src = Path(sys.argv[1])
    eng = json.loads(src.read_text())["train"]["engaged_ids"]
    split = json.loads(BS.OUT.read_text())
    stray = sorted(set(eng) - set(split["H1_train"]))
    if stray:
        raise SystemExit(f"engaged ids outside H1_train: {stray}")
    records = {r["id"]: r for r in json.loads(BS.SCENARIOS.read_text())["scenarios"]}
    d = draw(eng, records)
    out = {"rule": "ST_PLAN §5.0 T1.3 draw (author, 2026-10-19)", "seed": SEED,
           "strata": list(BS.STRATA), "n_engaged": len(eng),
           "inputs": {"engaged_ids_sha256": hashlib.sha256(
               json.dumps(sorted(eng)).encode()).hexdigest(),
               "split_v1.json": BS.sha256(BS.OUT)},
           "engaged": sorted(eng), **d}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.name}: gate {len(d['gate'])}, sweep {len(d['sweep'])}; quotas {d['quotas']}")


if __name__ == "__main__":
    main()
