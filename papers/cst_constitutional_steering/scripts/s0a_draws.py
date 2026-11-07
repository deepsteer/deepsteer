#!/usr/bin/env python3
"""S0a's seeded draws (zero GPU; committed before the pod; `S0A_BUILD.md`).

    s0a_draws.py capability     # data/s0a_capability_v1.json (D10: 400 GSM8K + 400 MMLU test)
    s0a_draws.py samples        # data/s0a_samples_v1.json (retest, Olmo's 10%, frontier 200)

Rollout ids are fixed before any rollout exists: ``{arm}|{set}|{condition}|{scenario_id}|{k}``
over the registered design (panel plain 4, user 2, operator 4, operator_pr 6; H1_held plain 8,
operator 8; cell A′ forced 8), so every sample is drawn from the plan, not from what generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/kdg_panel/scripts")]

CST = ROOT / "papers" / "cst_constitutional_steering"
ST = ROOT / "papers" / "st_live_reference"
ARMS = ("a0", "a0s")
PANEL_K = {"plain": 4, "user": 2, "operator": 4, "operator_pr": 6}
H1_K = {"plain": 8, "operator": 8}
FORCED_K = 8
SEED = 20261107


def design() -> dict[str, list[str]]:
    """Every rollout id of the registered design, per arm (scored sets: panel and H1)."""
    rec = json.loads((CST / "data/scenario_labels_v4_record.json").read_text())["labels"]
    panel = sorted(r["scenario_id"] for r in rec if r["set"] == "panel")
    h1 = sorted(r["scenario_id"] for r in rec if r["set"] == "h1")
    out = {}
    for arm in ARMS:
        ids = [f"{arm}|panel|{c}|{s}|{k}" for c, n in PANEL_K.items() for s in panel
               for k in range(n)]
        ids += [f"{arm}|h1|{c}|{s}|{k}" for c, n in H1_K.items() for s in h1 for k in range(n)]
        out[arm] = ids
    return out


def samples() -> dict:
    rec = {r["scenario_id"]: r for r in json.loads(
        (CST / "data/scenario_labels_v4_record.json").read_text())["labels"]}
    d = design()
    clear_op = [i for i in d["a0"] if "|panel|operator|" in i
                and rec[i.split("|")[3]]["stratum_of_record"] == "clear_refuse"]
    rng = random.Random(f"{SEED}|retest")
    retest = sorted(rng.sample(clear_op, 100))
    olmo = {arm: sorted(random.Random(f"{SEED}|olmo|{arm}").sample(ids, len(ids) // 10))
            for arm, ids in d.items()}
    frontier = sorted(random.Random(f"{SEED}|frontier|a0").sample(clear_op, 100)
                      + random.Random(f"{SEED}|frontier|a0s").sample(
                          [i.replace("a0|", "a0s|", 1) for i in clear_op], 100))
    return {"rule": "S0A_BUILD.md §2: rollout ids drawn from the registered design before the pod",
            "seed": SEED, "design_sizes": {a: len(v) for a, v in d.items()},
            "retest_a0_operator_clear": retest,
            "olmo_10pct": olmo,
            "frontier_200": frontier}


def capability() -> dict:
    """D10: 400 GSM8K and 400 MMLU test items, seeded, disjoint from ST's gate 200
    (``t15_capability_v1.json``) and pod 2's selection 100 (``st2qp_selection_v1.json``); indices
    and question sha256 only, as ST's sets."""
    from datasets import load_dataset

    used = set()
    for f in ("t15_capability_v1.json", "st2qp_selection_v1.json"):
        used |= {(i["dataset"], i["index"]) for i in json.loads((ST / "data" / f).read_text())
                 ["items"]}
    seed = f"{SEED}|cst_s0a_capability"
    out = {"rule": "CST_PLAN Amendment 1 D10: n 800, seeded, disjoint from ST's gate 200 and "
                   "selection 100", "seed": seed, "items": []}
    for name, cfg in (("openai/gsm8k", "main"), ("cais/mmlu", "all")):
        ds = load_dataset(name, cfg, split="test")
        free = [i for i in range(len(ds)) if (name, i) not in used]
        idx = sorted(random.Random(f"{seed}|{name}").sample(free, 400))
        out["items"] += [{"dataset": name, "config": cfg, "split": "test", "index": i,
                          "sha256": hashlib.sha256(ds[i]["question"].encode()).hexdigest()}
                         for i in idx]
        out[f"pool_{name}"] = len(free)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("capability", "samples"))
    a = ap.parse_args()
    f = CST / "data" / f"s0a_{a.step}_v1.json"
    f.write_text(json.dumps({"capability": capability, "samples": samples}[a.step](), indent=1)
                 + "\n")
    print(f"wrote {f}")
