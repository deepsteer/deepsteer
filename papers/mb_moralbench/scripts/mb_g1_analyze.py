"""MB-G1 analysis (MB_PLAN v0.5 §3-5): reads the tracked letters, scores them against the fetched
key, and writes aggregates only (``outputs/g1/<model>/g1_analysis.json``; no per-item scores).

Per cell (``t07`` of record, ``t10`` fork), set and foundation: the paper-comparison aggregate (a
miss scores 0), the parsed-only mean, R0 misses (stripped and unstripped), R1 and R1 - R0, R2's
expected score, and the match verdict against the paper row: inside the items-fixed 95% prediction
interval for a 5-repeat mean (+0.05 rounding), outside it, or "not attainable" under the key.
MB-A5: ``a_o``, ``a_c``, ``L``, ``K``, ``S`` with rollout-bootstrap CIs (§4). MB-A1 and MB-A3 reads.

Usage:
    python papers/mb_moralbench/scripts/mb_g1_analyze.py --model zephyr
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from mb_g0_audit import FOUNDATIONS, PAPER, PAPER_ROUNDING  # noqa: E402
from mb_g1_run import MODELS  # noqa: E402

from deepsteer.choice.ladder import ladder  # noqa: E402
from deepsteer.datasets.moralbench import MORALBENCH_SETS, load_moralbench  # noqa: E402
from deepsteer.stats.bootstrap import (  # noqa: E402
    k_repeat_prediction_interval,
    rollout_bootstrap_ci,
)

ANALYSIS_VERSION = "1.1.0"  # 1.1: variant cells (Amendment 1), R2 A-rate, MB-A7 split
REGISTERED = ("zephyr", "gemma", "llama2_70b")  # rows of record; others are Amendment 1 variants
N_DRAWS, N_BOOT = 10_000, 2_000
NOT_ATTAINABLE_REASON = {
    ("Zephyr", "MFQ_30_compare", "ingroup"): "MB-A3: below the tied item's floor when every "
    "reply parses; reachable only through misses scored 0",
    ("LLaMA-2", "6_concepts", "purity"): "MB-A2: above the key's oracle",
}


def letters(rows: list[dict], items: list, field: str) -> np.ndarray:
    """``[n_items, n_rollouts]`` object array of labels (None = miss), items in ``items`` order."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["set"], r["item_id"])][r["rollout"]] = r[field]
    n = max(len(v) for v in by.values())
    return np.array([[by[(it.set_name, it.item_id)].get(k) for k in range(n)] for it in items],
                    dtype=object)


def scores(lab: np.ndarray, items: list) -> np.ndarray:
    return np.array([[it.key[x] if x is not None else np.nan for x in row]
                     for it, row in zip(items, lab)], dtype=float)


def a_indicator(lab: np.ndarray) -> np.ndarray:
    return np.where(lab == None, np.nan, (lab == "A").astype(float))  # noqa: E711


def cell_summary(rows: dict, items: list, model_row: str, match_label: str) -> dict:
    out = {}
    for s in MORALBENCH_SETS:
        its = [it for it in items if it.set_name == s]
        lab0 = letters(rows["r0"], its, "r0")
        labu = letters(rows["r0"], its, "r0_unstripped")
        lab1 = letters(rows["r0"], its, "r1")
        sc0, sc1 = scores(lab0, its), scores(lab1, its)
        groups = [f for f in FOUNDATIONS if any(it.foundation_label == f for it in its)]
        prow = PAPER[s][model_row]
        paper = dict(zip(FOUNDATIONS + ["TOTAL"], prow[:6] + [prow[6]]))
        res = {}
        for f in groups + ["TOTAL"]:
            m = np.array([f == "TOTAL" or it.foundation_label == f for it in its])
            g_items = [it for it, keep in zip(its, m) if keep]
            band = ladder(g_items)
            x0, x1 = sc0[m], sc1[m]
            lo, hi, _ = k_repeat_prediction_interval(x0, 5, n_draws=N_DRAWS, seed=0)
            lo, hi = lo - PAPER_ROUNDING, hi + PAPER_ROUNDING
            agg = float(np.nan_to_num(x0, nan=0.0).mean(axis=1).sum())
            parsed = np.nanmean(x0, axis=1)
            pv = paper.get(f)
            entry = {
                "n_items": int(m.sum()), "paper": pv,
                "r0_miss0": round(agg, 3),
                "r0_parsed_only": round(float(np.nansum(parsed)), 3),
                "items_never_parsed": int(np.isnan(parsed).sum()),
                "r0_misses": int(np.isnan(x0).sum()),
                "r0_unstripped_misses": int(sum(v is None for v in labu[m].ravel())),
                "n_replies": int(x0.size),
                "r1_miss0": round(float(np.nan_to_num(x1, nan=0.0).mean(axis=1).sum()), 3),
                "pi95_5rep": [round(lo, 3), round(hi, 3)],
                "band": [band["worst"], band["oracle"]],
                "always_A": band["always_A"], "always_B": band["always_B"],
            }
            entry["r1_minus_r0"] = round(entry["r1_miss0"] - entry["r0_miss0"], 3)
            if pv is not None:
                key = (model_row, s, f)
                outside = not (band["worst"] - PAPER_ROUNDING <= pv
                               <= band["oracle"] + PAPER_ROUNDING)
                if key in NOT_ATTAINABLE_REASON or outside:
                    entry["verdict"] = "not_attainable"
                    entry["not_attainable_reason"] = NOT_ATTAINABLE_REASON.get(key, "outside band")
                else:
                    entry["verdict"] = match_label if lo <= pv <= hi else "does_not_replicate"
            res[f] = entry
        out[s] = res
    return out


def r2_summary(r2: list[dict], items: list) -> dict:
    by = {(r["set"], r["item_id"]): r for r in r2 if r["order"] == "original"}
    out = {}
    for s in MORALBENCH_SETS:
        its = [it for it in items if it.set_name == s]
        res = {}
        groups = [f for f in FOUNDATIONS if any(it.foundation_label == f for it in its)]
        for f in groups + ["TOTAL"]:
            g = [it for it in its if f == "TOTAL" or it.foundation_label == f]
            exp = sum(by[(it.set_name, it.item_id)]["p_A"] * it.key["A"]
                      + by[(it.set_name, it.item_id)]["p_B"] * it.key["B"] for it in g)
            res[f] = {"expected": round(exp, 3),
                      "a_rate": round(float(np.mean(
                          [by[(it.set_name, it.item_id)]["p_A"] for it in g])), 4),
                      "mean_label_mass": round(float(np.mean(
                          [by[(it.set_name, it.item_id)]["label_mass"] for it in g])), 4)}
        out[s] = res
    return out


def acquiescence(rows_o: list[dict], rows_c: list[dict], items: list) -> dict:
    out = {}
    for s in MORALBENCH_SETS:
        its = [it for it in items if it.set_name == s]
        ao_x = a_indicator(letters(rows_o, its, "r0"))
        ac_x = a_indicator(letters(rows_c, its, "r0"))
        delta = np.array([it.key["A"] - it.key["B"] for it in its])

        def a_rate(x):
            return float(np.nanmean(x))

        def stat_l(xo, xc):
            return (a_rate(xo) + a_rate(xc)) / 2 - 0.5

        def stat_k(xo, xc):
            return (a_rate(xo) - a_rate(xc)) / 2

        def stat_s(xo):
            p = np.nanmean(xo, axis=1)
            ok = ~np.isnan(p)
            return float(((p[ok] - np.nanmean(xo)) * delta[ok]).sum())

        ests = {
            "a_o": rollout_bootstrap_ci(a_rate, ao_x, n_boot=N_BOOT, seed=1),
            "a_c": rollout_bootstrap_ci(a_rate, ac_x, n_boot=N_BOOT, seed=2),
            "L": rollout_bootstrap_ci(stat_l, ao_x, ac_x, n_boot=N_BOOT, seed=3),
            "K": rollout_bootstrap_ci(stat_k, ao_x, ac_x, n_boot=N_BOOT, seed=4),
            "S": rollout_bootstrap_ci(stat_s, ao_x, n_boot=N_BOOT, seed=5),
        }
        res = {k: [round(v, 4) for v in t] for k, t in ests.items()}
        s_lo, s_hi = ests["S"][1], ests["S"][2]
        res["verdict"] = "R_acquiescence" if s_lo <= 0 <= s_hi else "R_content"
        res["parsed_o"] = int(np.isfinite(ao_x).sum())
        res["parsed_c"] = int(np.isfinite(ac_x).sum())
        out[s] = res
    return out


def mb_a7_split(rows: list[dict]) -> dict:
    """R1's letter among R0 hits against among R0 misses, per set (ANOMALIES MB-A7)."""
    out = {}
    for s in MORALBENCH_SETS:
        rs = [r for r in rows if r["set"] == s]
        hit = defaultdict(int)
        miss = defaultdict(int)
        for r in rs:
            (hit if r["r0"] is not None else miss)[str(r["r1"])] += 1
        out[s] = {"r1_among_r0_hits": dict(hit), "r1_among_r0_misses": dict(miss)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--root", type=Path, default=REPO / "papers" / "mb_moralbench")
    ap.add_argument("--sub", default="g1", help="results folder under outputs/ (g2 for the pod)")
    args = ap.parse_args()
    d = args.root / "outputs" / args.sub / args.model
    _, items = load_moralbench()
    load = lambda name: [json.loads(x) for x in (d / name).read_text().splitlines()]  # noqa: E731
    row = MODELS[args.model]["paper_row"]
    label = "replicates" if args.model in REGISTERED else f"matches_under_{args.model}"
    cells = [c for c in ("t07", "t10") if (d / f"{c}.jsonl").exists()]
    out = {
        "metadata": {"analysis_version": ANALYSIS_VERSION, "date": dt.date.today().isoformat(),
                     "model_key": args.model, "paper_row": row, "n_draws": N_DRAWS,
                     "n_boot": N_BOOT, "rounding": PAPER_ROUNDING,
                     "registered_configuration": args.model in REGISTERED,
                     "match_rule": "items fixed; 95% PI of a 5-repeat mean, miss scores 0",
                     "results_folder": args.sub},
        "cells": {c: cell_summary({"r0": load(f"{c}.jsonl")}, items, row, label) for c in cells},
        "r2": r2_summary(load("r2.jsonl"), items),
        "mb_a7": mb_a7_split(load("t07.jsonl")),
    }
    if (d / "cb07.jsonl").exists():
        out["mb_a5"] = acquiescence(load("t07.jsonl"), load("cb07.jsonl"), items)
    (d / "g1_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for c in cells:
        for s, res in out["cells"][c].items():
            line = " ".join(f"{f}:{e['r0_miss0']}({e.get('verdict', '-')[:4]})"
                            for f, e in res.items())
            print(f"{c} {s}: {line}")
    for s, r in out.get("mb_a5", {}).items():
        print(f"A5 {s}: a_o {r['a_o']} a_c {r['a_c']} L {r['L']} K {r['K']} "
              f"S {r['S']} {r['verdict']}")


if __name__ == "__main__":
    main()
