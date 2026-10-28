#!/usr/bin/env python3
"""T1.5's specificity set and capability slice (ST_PLAN §3, Amendment 19), drawn once.

    python3 papers/st_live_reference/scripts/build_t15_sets.py

- ``data/t15_specificity_v1.json``: 60 of the panel primaries whose external label of record
  (``external_label.right_option_id``) names the rule-following (``consistent``) option, seed
  20261021, stratified by family (largest-remainder quotas).
- ``data/t15_capability_v1.json``: 100 GSM8K test items (``openai/gsm8k``, ``main``) and 100 MMLU
  test items (``cais/mmlu``, ``all``), seed 20261021; stream indices and question sha256 only (the
  pod re-streams, verifies and scores against the datasets' own answers).
- ``python3 build_t15_sets.py panel_half``: ``data/t15_panel_half_v1.json`` (Amendment 20 item 6).
- ``python3 build_t15_sets.py lbeh2_pool``: ``data/st_lbeh2_pool_v1.json``, L-beh-2's rule-right
  training pool (Amendment 31 item 3 and its addition).
- ``python3 build_t15_sets.py st2qp_selection``: ``data/st2qp_selection_v1.json``, pod 2's held-out
  checkpoint-selection set (Amendment 57 item 5), 50 GSM8K and 50 MMLU disjoint from the gate's 200.
- ``python3 build_t15_sets.py engaged <t11 dir>``: ``data/t15_engaged_held_v1.json``, T1.1's 38
  brake-engaged H1_held ids copied from ``select/engaged.json`` (no new draw), so an HF fallback
  generates ΔP's scenarios first (Amendment 18 item 4's scope).
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


def engaged(t11: Path) -> dict:
    f = t11 / "select" / "engaged.json"
    h = json.loads(f.read_text())["held"]
    if len(h["engaged_ids"]) != h["n_engaged"]:
        raise SystemExit("engaged.json is inconsistent")
    return {"rule": f"T1.1's held brake-engaged subset ({h['rule']}); ST_PLAN Amendment 1",
            "n": h["n_engaged"], "ids": sorted(h["engaged_ids"]),
            "source_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}


def lbeh2_pool() -> dict:
    """Amendment 31 item 3 and its addition: the rule-right primaries outside the specificity set
    and outside the panel guard's half, L-beh-2's rule-right training pool."""
    import screen_lib as L

    from deepsteer.kdg.schema import load_scenario_dir

    allp, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                     "*_scenarios_*.json"))))
    pool = {s.id: s.family for s in L.panel_586(allp)
            if s.role == "primary" and (s.external_label or {}).get("right_option_id")
            and {o.option_id: o.norm_status for o in s.options}.get(
                s.external_label["right_option_id"]) == "consistent"}
    spec = set(json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"])
    half = set(json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"])
    ids = sorted(set(pool) - spec - half)
    return {"rule": "ST_PLAN Amendment 31 item 3 and its addition: rule-right primaries outside "
                    "the specificity set and the panel guard's half", "n_rule_right": len(pool),
            "per_family": dict(sorted(Counter(pool[i] for i in ids).items())), "ids": ids,
            "inputs_sha256": {f: hashlib.sha256((DATA / f).read_bytes()).hexdigest() for f in
                              ("t15_specificity_v1.json", "t15_panel_half_v1.json")}}


def st2qp_selection() -> dict:
    """Amendment 57 item 5: pod 2's checkpoint-selection set, 50 GSM8K and 50 MMLU test items
    disjoint from the gate's 200 (``t15_capability_v1.json``), seeded; indices and question sha256
    only, as the gate's set."""
    from datasets import load_dataset

    gate = json.loads((DATA / "t15_capability_v1.json").read_text())
    used = {(i["dataset"], i["index"]) for i in gate["items"]}
    seed = "20261028|st2qp_selection"
    out = {"rule": "ST_PLAN Amendment 57 item 5: held out from the gate's 200; never the gate's "
                   "capability read", "seed": seed, "items": [],
           "gate_set_sha256": hashlib.sha256((DATA / "t15_capability_v1.json").read_bytes())
           .hexdigest()}
    for name, cfg in (("openai/gsm8k", "main"), ("cais/mmlu", "all")):
        ds = load_dataset(name, cfg, split="test")
        free = [i for i in range(len(ds)) if (name, i) not in used]
        idx = sorted(random.Random(f"{seed}|{name}").sample(free, 50))
        out["items"] += [{"dataset": name, "config": cfg, "split": "test", "index": i,
                          "sha256": _sha(ds[i]["question"])} for i in idx]
    return out


def main() -> None:
    if sys.argv[1:] == ["st2qp_selection"]:
        d = st2qp_selection()
        (DATA / "st2qp_selection_v1.json").write_text(json.dumps(d, indent=1) + "\n")
        print(f">> st2qp selection: {dict(Counter(i['dataset'] for i in d['items']))}")
        return
    if sys.argv[1:] == ["lbeh2_pool"]:
        d = lbeh2_pool()
        (DATA / "st_lbeh2_pool_v1.json").write_text(json.dumps(d, indent=1) + "\n")
        print(f">> L-beh-2 rule-right pool: {len(d['ids'])} ({d['per_family']})")
        return
    if sys.argv[1:2] == ["engaged"]:
        e = engaged(Path(sys.argv[2]))
        (DATA / "t15_engaged_held_v1.json").write_text(json.dumps(e, indent=1) + "\n")
        print(f">> engaged H1_held: {e['n']}")
        return
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
