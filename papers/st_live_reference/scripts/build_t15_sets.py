#!/usr/bin/env python3
"""T1.5's specificity set and capability slice (ST_PLAN §3, Amendment 19), drawn once.

    python3 papers/st_live_reference/scripts/build_t15_sets.py

- ``data/t15_specificity_v1.json``: 60 of the panel primaries whose external label of record
  (``external_label.right_option_id``) names the rule-following (``consistent``) option, seed
  20261021, stratified by family (largest-remainder quotas).
- ``data/t15_capability_v1.json``: 100 GSM8K test items (``openai/gsm8k``, ``main``) and 100 MMLU
  test items (``cais/mmlu``, ``all``), seed 20261021; stream indices and question sha256 only (the
  pod re-streams, verifies and scores against the datasets' own answers).
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
SEED, N_SPEC, N_CAP = 20261021, 60, 100


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    tot = sum(counts.values())
    exact = {k: n * c / tot for k, c in counts.items()}
    q = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - q[k]), k))[: n - sum(q.values())]:
        q[k] += 1
    return q


def specificity() -> dict:
    import screen_lib as L

    from deepsteer.kdg.schema import load_scenario_dir

    allp, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                     "*_scenarios_*.json"))))
    pool = []
    for s in L.panel_586(allp):
        ro = (s.external_label or {}).get("right_option_id")
        ns = {o.option_id: o.norm_status for o in s.options}
        if s.role == "primary" and ro and ns.get(ro) == "consistent":
            pool.append(s)
    by: dict[str, list[str]] = {}
    for s in pool:
        by.setdefault(s.family, []).append(s.id)
    q = quotas({f: len(v) for f, v in by.items()}, N_SPEC)
    ids = []
    for f in sorted(by):
        m = sorted(by[f])
        random.Random(f"{SEED}|{f}").shuffle(m)
        ids += m[: q[f]]
    return {"rule": "ST_PLAN Amendment 19 item 3", "seed": SEED, "pool": len(pool),
            "per_family_pool": {f: len(v) for f, v in sorted(by.items())},
            "quotas": dict(sorted(q.items())), "ids": sorted(ids)}


def panel_half() -> dict:
    """Amendment 20 item 6: the panel guard's 50% subsample, 293 of the 586, seed 20261021,
    stratified by family (largest-remainder quotas)."""
    import screen_lib as L

    from deepsteer.kdg.schema import load_scenario_dir

    allp, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                     "*_scenarios_*.json"))))
    by: dict[str, list[str]] = {}
    for s in L.panel_586(allp):
        by.setdefault(s.family, []).append(s.id)
    q = quotas({f: len(v) for f, v in by.items()}, 293)
    ids = []
    for f in sorted(by):
        m = sorted(by[f])
        random.Random(f"{SEED}|half|{f}").shuffle(m)
        ids += m[: q[f]]
    return {"rule": "ST_PLAN Amendment 20 item 6", "seed": SEED, "n": len(ids),
            "quotas": dict(sorted(q.items())), "ids": sorted(ids)}


def capability() -> dict:
    from datasets import load_dataset

    out = {"rule": "ST_PLAN Amendment 19 item 4", "seed": SEED, "items": []}
    for name, cfg, field in (("openai/gsm8k", "main", "question"), ("cais/mmlu", "all",
                                                                   "question")):
        ds = load_dataset(name, cfg, split="test")
        idx = sorted(random.Random(f"{SEED}|{name}").sample(range(len(ds)), N_CAP))
        out["items"] += [{"dataset": name, "config": cfg, "split": "test", "index": i,
                          "sha256": _sha(ds[i][field])} for i in idx]
    out["fallback_100"] = "the first 50 of each dataset in this order"
    return out


def main() -> None:
    if sys.argv[1:] == ["panel_half"]:
        h = panel_half()
        (DATA / "t15_panel_half_v1.json").write_text(json.dumps(h, indent=1) + "\n")
        print(f">> panel half: {h['n']} (quotas {h['quotas']})")
        return
    s = specificity()
    (DATA / "t15_specificity_v1.json").write_text(json.dumps(s, indent=1) + "\n")
    c = capability()
    (DATA / "t15_capability_v1.json").write_text(json.dumps(c, indent=1) + "\n")
    print(f">> specificity: {len(s['ids'])} of {s['pool']} (quotas {s['quotas']}); capability: "
          f"{dict(Counter(i['dataset'] for i in c['items']))}")


if __name__ == "__main__":
    main()
