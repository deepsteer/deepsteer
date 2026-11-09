#!/usr/bin/env python3
"""CST Amendment 22 items 2-4: the headline's sensitivity to the labels (descriptive).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/s0a_label_sensitivity.py

The operator cell's ordered-option share for A0 and A0s and their scenario-paired Δ
(`s0a_reads.paired`, the headline's own estimator and units: rollouts with an action, capped
excluded) on: the clear items of record (458), the family splits (68), the boundary items (57), all
586 panel items with no labels, and each family's own clear set. When Claude's regrouped labels
exist (`data/scenario_labels_v4_claude_permuted.json`, Amendment 22 item 1), also: item-level
agreement between the two groupings per set (panel, H1_held), the stable subset (clear of record and
clear under Claude's regrouped labels), its Δ, and the trigger (item 3): random subsets of the 458
at the stable subset's size, 2,000 draws, the null's central 95% of Δ. Writes
`outputs/s0a_label_sensitivity.json`.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402
import s0a_reads as R  # noqa: E402

CST = HERE.parent
SEED = "20261109|cst_a22_null"
N_NULL = 2000


def ordered_units(lab: dict) -> dict[str, dict[str, list[int]]]:
    """Arm -> scenario -> ordered-option indicator per operator-cell rollout with an action."""
    out = {}
    for a in ("a0", "a0s"):
        u: dict[str, list[int]] = defaultdict(list)
        for r in A.load_jsonl(f"gen/{a}/panel_operator_*.jsonl"):
            if r["condition"] != "operator" or r["hit_cap"] or r["option"] is None:
                continue
            u[r["scenario_id"]].append(int(r["option"] == lab["record"][r["scenario_id"]]
                                           ["ordered_option"]))
        out[a] = dict(u)
    return out


def read(units: dict, ids: set[str]) -> dict:
    return R.paired({s: v for s, v in units["a0"].items() if s in ids},
                    {s: v for s, v in units["a0s"].items() if s in ids})


def regroup(lab: dict, units: dict) -> dict | None:
    f = CST / "data/scenario_labels_v4_claude_permuted.json"
    if not f.exists():
        return None
    import scenario_labels as L

    from deepsteer.constitutions import TenetSet

    tenets = TenetSet.load(CST / "data/tenets_C_anth_v2.json")
    ranks = L._ranks(tenets)  # noqa: SLF001
    _, by, ordered = L._ordered_and_opts()  # noqa: SLF001
    perm = {r["scenario_id"]: r for r in json.loads(f.read_text())["labels"]}
    out: dict = {}
    for set_ in ("panel", "h1"):
        ids = [s for s, v in lab["record"].items() if v["set"] == set_ and s in perm]
        agree, clear_both, clear_orig = 0, [], []
        for s in ids:
            sc, _ = by[s]
            p = L.label_rule_v4(perm[s], ordered[s], ranks, [o.option_id for o in sc.options])
            o = lab["record"][s]["per_family"]["claude"]
            agree += p["direction"] == o["direction"]
            if lab["record"][s]["stratum_of_record"] == "clear_refuse":
                clear_orig.append(s)
                if p["stratum"] == "clear_refuse":
                    clear_both.append(s)
        out[set_] = {"n": len(ids), "direction_agreement": agree / max(len(ids), 1),
                     "clear_of_record": len(clear_orig), "stay_clear_under_regroup":
                     len(clear_both), "stable_ids": sorted(clear_both)}
    stable = set(out["panel"]["stable_ids"])
    clear = sorted(s for s, v in lab["record"].items() if v["stratum_of_record"] == "clear_refuse"
                   and v["set"] == "panel")
    out["stable_subset"] = read(units, stable)
    rng = random.Random(SEED)
    null = sorted(read(units, set(rng.sample(clear, len(stable))))["delta"]
                  for _ in range(N_NULL))
    lo, hi = null[int(0.025 * N_NULL)], null[int(0.975 * N_NULL) - 1]
    d = out["stable_subset"]["delta"]
    out["trigger"] = {"null_central_95": [lo, hi], "stable_delta": d,
                      "outside": bool(d < lo or d > hi), "n_null": N_NULL, "seed": SEED,
                      "rule": "Amendment 22 item 3: outside -> ledger and the author; inside -> "
                              "CST-A19 closes for the headline with the agreement rate stated"}
    return out


def main() -> None:
    lab = A.labels()
    units = ordered_units(lab)
    rec = lab["record"]
    panel = {s for s, v in rec.items() if v["set"] == "panel"}
    sets = {"clear_of_record": {s for s in panel if rec[s]["stratum_of_record"] == "clear_refuse"},
            "family_split": {s for s in panel if rec[s]["stratum_of_record"] == "family_split"},
            "boundary": {s for s in panel if rec[s]["stratum_of_record"] == "boundary"},
            "all_586_no_labels": panel,
            "claude_clear": {s for s in panel if rec[s]["per_family"]["claude"]["stratum"]
                             == "clear_refuse"},
            "codex_clear": {s for s in panel if rec[s]["per_family"]["codex"]["stratum"]
                            == "clear_refuse"}}
    res = {"rule": "CST Amendment 22 items 2-4 (descriptive)",
           "by_set": {k: read(units, v) for k, v in sets.items()}, "regroup": regroup(lab, units)}
    # most probable failure: the clear set's row is not the headline's (another estimator or unit)
    ref = json.loads((CST / "outputs/s0a_reads.json").read_text())["actions"][
        "panel|operator|clear_refuse"]["ordered"]
    assert abs(res["by_set"]["clear_of_record"]["delta"] - ref["delta"]) < 1e-9, ref
    (CST / "outputs/s0a_label_sensitivity.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["by_set"].items():
        print(f"{k:20s} n {v['scenarios']:3d}  A0 {v['a0']:.3f}  A0s {v['a0s']:.3f}  "
              f"Δ {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    if res["regroup"]:
        g = res["regroup"]
        for s in ("panel", "h1"):
            print(s, {k: v for k, v in g[s].items() if k != "stable_ids"})
        print("stable", g["stable_subset"], "trigger", g["trigger"])


if __name__ == "__main__":
    main()
