#!/usr/bin/env python3
"""S0a's descriptive and ST-comparability reads (zero GPU): action shares per cell, ΔP on H1's 82
clear items, the capability Δ under both parsers (Amendment 13 item 1), the recall margins per
token, and Olmo's disagreement with the scorer of record. Writes outputs/s0a_reads.json. Paired
A0s − A0 differences use a scenario bootstrap (2,000 draws, seeded)."""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402

CST = A.CST
B = 2000


def paired(units_a: dict, units_b: dict, seed: int = 20261108) -> dict:
    """A0s − A0 of per-scenario means, scenario-paired bootstrap."""
    sids = sorted(set(units_a) & set(units_b))
    def d(ids):
        return (statistics.mean(statistics.mean(units_b[s]) for s in ids)
                - statistics.mean(statistics.mean(units_a[s]) for s in ids))
    rng = random.Random(seed)
    dr = sorted(d([rng.choice(sids) for _ in sids]) for _ in range(B))
    def m(u):
        return statistics.mean(statistics.mean(u[s]) for s in sids)
    return {"scenarios": len(sids), "a0": m(units_a), "a0s": m(units_b),
            "delta": d(sids), "ci95": [dr[int(0.025 * B)], dr[int(0.975 * B) - 1]]}


def actions(lab: dict) -> dict:
    out = {}
    gen = {a: A.load_jsonl(f"gen/{a}/*.jsonl") for a in ("a0", "a0s")}
    for set_, cond in (("panel", "plain"), ("panel", "user"), ("panel", "operator"),
                       ("panel", "operator_pr"), ("h1", "plain"), ("h1", "operator")):
        for stratum in ("clear_refuse", "boundary", "family_split"):
            per = {}
            for a in ("a0", "a0s"):
                u = defaultdict(lambda: defaultdict(list))
                for r in gen[a]:
                    if r["set"] != set_ or r["condition"] != cond:
                        continue
                    rec = lab["record"][r["scenario_id"]]
                    if rec["stratum_of_record"] != stratum:
                        continue
                    act = r["option"] if not r["hit_cap"] else None
                    if act is None:
                        continue
                    sid = r["scenario_id"]
                    u["directed"][sid].append(int(act == rec["directed_option_of_record"]))
                    u["ordered"][r["scenario_id"]].append(int(act == rec["ordered_option"]))
                    u["o3"][r["scenario_id"]].append(int(act == "o3"))
                per[a] = u
            if not per["a0"]["directed"]:
                continue
            out[f"{set_}|{cond}|{stratum}"] = {k: paired(per["a0"][k], per["a0s"][k])
                                               for k in ("directed", "ordered", "o3")}
    return out


def capability() -> dict:
    rows = {a: {(r["dataset"], r["index"]): r for r in map(json.loads, open(
        A.S0A / f"cap/{a}.jsonl"))} for a in ("a0", "a0s")}
    out = {}
    for key in ("correct", "correct_st"):
        for ds in ("all", "openai/gsm8k", "cais/mmlu"):
            ids = [k for k in rows["a0"] if (ds == "all" or k[0] == ds) and k in rows["a0s"]]
            ua = {k: [int(rows["a0"][k][key])] for k in ids}
            ub = {k: [int(rows["a0s"][k][key])] for k in ids}
            out[f"{key}|{ds}"] = paired(ua, ub)
    return out


def recall() -> dict:
    by = defaultdict(dict)
    for r in map(json.loads, open(A.S0A / "recall/a0.jsonl")):
        by[r["id"]][r["variant"]] = r["sum_logprob"] / r["n"]  # per token
    out = {}
    for st in ("C_anth", "C_spec", "public_domain"):
        m = [v["true"] - v["p1"] for k, v in by.items() if k.startswith(st)]
        fl = [v["p1"] - v["p2"] for k, v in by.items() if k.startswith(st)]
        out[st] = {"n": len(m), "margin_per_token_median": statistics.median(m),
                   "floor_per_token_median": statistics.median(fl),
                   "share_margin_above_floor_q95": sum(x > sorted(map(abs, fl))[int(0.95 * len(fl))]
                                                       for x in m) / len(m)}
    return out


def olmo() -> dict:
    out = {}
    for a in ("a0", "a0s"):
        g = {r["rollout_id"]: r for r in A.load_jsonl(f"score/{a}/gpt_oss_120b/blind_*.jsonl")}
        o = A.load_jsonl(f"score/{a}/olmo31_32b_instruct/blind_*.jsonl")
        dis = sum(x["concluded_option"] != g[x["rollout_id"]]["concluded_option"] for x in o)
        out[a] = {"n": len(o), "concluded_disagreement": dis / len(o)}
    return out


if __name__ == "__main__":
    lab = A.labels()
    res = {"actions": actions(lab), "capability": capability(), "recall_a0": recall(),
           "olmo_10pct": olmo()}
    (CST / "outputs" / "s0a_reads.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["actions"].items():
        if "clear" in k:
            print(k, {m: (round(x["a0"], 3), round(x["a0s"], 3), round(x["delta"], 3),
                          [round(c, 3) for c in x["ci95"]]) for m, x in v.items()})
    for k, v in res["capability"].items():
        print("cap", k, round(v["a0"], 3), round(v["a0s"], 3), round(v["delta"], 4),
              [round(c, 4) for c in v["ci95"]])
    print("recall", {k: {m: round(x, 3) for m, x in v.items()}
                     for k, v in res["recall_a0"].items()})
    print("olmo", res["olmo_10pct"])
