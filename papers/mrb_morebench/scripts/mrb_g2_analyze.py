"""MRB-G2 analysis (MRB_PLAN v0.7): the theory set on Qwen3-30B-A3B -> ``g2_analysis.json``.

    mrb_g2_analyze.py [--out papers/mrb_morebench/outputs/g2]

Per field (trace, response), on the 150 rows of the paper-faithful generation (k0) under
upstream's substring credit (exact-token credit beside it): Overall and Overall-LC (the paper's
Hard: Overall / mean length x 1000) against the paper's theory tables, with G1's registered
match rule (95% PI of a one-generation mean, k = 2 resampled, rows fixed, +/- 0.05) and G1's
sensitivity interval (widened by the deployment term the cross-check measured on Qwen's fields);
the five per-theory cells (30 rows each) under the same rule, as secondary cells; the two-rung
ladder; the verdict rates; and a table-arithmetic audit of the paper's own rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mrb_g1_analyze as A  # noqa: E402

from deepsteer.datasets.morebench import load_morebench  # noqa: E402

ANALYSIS_VERSION = "1.0.0"
THEORY_ORDER = ("Gauthierian Contractarianism", "Scanlonian Contractualism", "Act Utilitarianism",
                "Aristotelian Virtue Ethics", "Kantian Deontology")
# arXiv:2510.16380v2, table/model_resp_on_benchmark_theory.tex (the two MoReBench-Theory tables,
# read from the v2 source on 2026-10-31), row "30B-A3B": Gauthierian, Scanlonian, Benthamite,
# Aristotelian, Kantian, Overall, Length (chars), Overall-LC
PAPER_THEORY = {"trace": (70.2, 71.6, 76.7, 74.7, 75.4, 73.7, 4766, 15.5),
                "response": (69.2, 74.4, 77.0, 78.5, 79.2, 75.7, 7396, 10.2)}
# MRB_PLAN v0.6 R2: the larger absolute end of the OpenRouter - pod Regular CI on Qwen's fields
DEPLOYMENT_DELTA = {"trace": 3.40, "response": 2.03}
MODEL = "qwen3_30b"


def pi(score: np.ndarray, length: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    n, k = score.shape
    pick = rng.integers(0, k, (A.N_DRAWS, n))
    rows = np.arange(n)[None, :]
    reg = score[rows, pick].mean(1)
    return reg, reg / length[rows, pick].mean(1) * 1000


def verdict(dist: np.ndarray, paper: float, delta: float = 0.0) -> dict:
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    out = {"paper": paper, "pi95": [round(lo, 2), round(hi, 2)],
           "verdict": ("replicates" if lo - A.ROUNDING <= paper <= hi + A.ROUNDING else "misses")}
    if delta:
        w = [lo - A.ROUNDING - delta, hi + A.ROUNDING + delta]
        out["sensitivity_interval"] = [round(w[0], 2), round(w[1], 2)]
        out["sensitivity_verdict"] = ("replicates within deployment noise"
                                      if w[0] <= paper <= w[1] else "misses")
    return out


def table_audit() -> dict:
    out = {}
    for f, row in PAPER_THEORY.items():
        out[f] = {"overall_from_theories": round(float(np.mean(row[:5])), 2), "overall": row[5],
                  "lc_from_overall_and_length": round(row[5] / row[6] * 1000, 1), "lc": row[7]}
        out[f]["consistent"] = (abs(out[f]["overall_from_theories"] - row[5]) <= 0.05
                                and out[f]["lc_from_overall_and_length"] == row[7])
    return out


def main_from(out: Path) -> dict:
    rng = np.random.default_rng(A.SEED)
    items = load_morebench(subset="theory")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    rows = A.read_jsonl(out / "judge" / "shard0.jsonl")
    us = A.unit_scores(rows, weights)
    gens = {r["job_id"]: r for r in A.read_jsonl(out / "gen" / f"{MODEL}.jsonl")}
    ks = sorted({int(g.split("/")[3][1:]) for g in gens})
    items = [it for it in items if f"{MODEL}/default/{it.scenario_id}/k0" in gens]  # VALIDATE: 15

    fields = {}
    for f in ("trace", "response"):
        cell = PAPER_THEORY[f]
        per_credit = {}
        for c in A.CREDITS:
            sc = np.array([[us[f"{MODEL}/default/{it.scenario_id}/k{k}/{f}"][c] for k in ks]
                           for it in items])
            ln = np.array([[gens[f"{MODEL}/default/{it.scenario_id}/k{k}"][f"{f}_chars"]
                            for k in ks] for it in items], dtype=float)
            reg_d, hard_d = pi(sc, ln, rng)
            ov, l0 = float(sc[:, 0].mean()), float(ln[:, 0].mean())
            theories = {}
            for t, paper in zip(THEORY_ORDER, cell[:5]):
                m = np.array([it.theory == t for it in items])
                if not m.any():
                    continue
                d, _ = pi(sc[m], ln[m], rng)
                theories[t] = {"n": int(m.sum()), "ours_k0": round(float(sc[m, 0].mean()), 2),
                               **verdict(d, paper)}
            per_credit[c] = {
                "n_rows": len(items), "k": len(ks), "overall_k0": round(ov, 2),
                "overall_k0_ci95": A.scenario_ci(sc[:, 0], rng), "length_k0": round(l0),
                "overall_lc_k0": round(ov / l0 * 1000, 2), "paper_length": cell[6],
                "scenario_sd_k0": round(float(sc[:, 0].std(ddof=1)), 2),
                "generation_sd_within": round(float(np.sqrt(np.mean(sc.var(1, ddof=1)))), 2),
                "match_overall": verdict(reg_d, cell[5], DEPLOYMENT_DELTA[f]),
                "match_overall_lc": verdict(hard_d, cell[7]),
                "theories": theories}
        fields[f] = per_credit

    canned = sorted({u.split("/")[1] for u in us if u.startswith("canned/")})
    ladder = {c: {name: round(float(np.mean([us[f"canned/{name}/{it.scenario_id}"][c]
                                             for it in items])), 2) for name in canned}
              for c in A.CREDITS}
    types = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        key = "canned" if r["unit"].startswith("canned/") else r["unit"].split("/")[-1]
        types[key][0] += 1
        types[key][1] += r["verdict"] is None
        types[key][2] += r["n_tokens"]
    verdicts = {k: {"calls": v[0], "no_verdict": v[1], "no_verdict_pct": round(100 * v[1] / v[0], 2),
                    "tokens_per_call": round(v[2] / v[0])} for k, v in sorted(types.items())}
    g = list(gens.values())
    res = {"metadata": {"analysis_version": ANALYSIS_VERSION, "date": dt.date.today().isoformat(),
                        "plan": "MRB_PLAN v0.7", "judge_rows": len(rows), "units": len(us),
                        "units_incomplete": sum(not u["complete"] for u in us.values()),
                        "n_draws": A.N_DRAWS, "seed": A.SEED, "deployment_delta": DEPLOYMENT_DELTA},
           "paper_table_audit": table_audit(),
           "generation": {"n": len(g), "final_reached": sum(r["final_reached"] for r in g),
                          "hit_cap": sum(r["finish_reason"] == "length" for r in g),
                          "empty_trace": sum(r["trace_chars"] == 0 for r in g)},
           "verdicts_by_type": verdicts, "fields": fields, "ladder": ladder}
    (out / "g2_analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    for f, v in fields.items():
        b = v["credit_substring"]
        print(f"{f:9s} Overall {b['overall_k0']:6.2f} {b['overall_k0_ci95']} (paper "
              f"{b['match_overall']['paper']}) {b['match_overall']['verdict']} PI "
              f"{b['match_overall']['pi95']} | LC {b['overall_lc_k0']:6.2f} (paper "
              f"{b['match_overall_lc']['paper']}) {b['match_overall_lc']['verdict']} | len "
              f"{b['length_k0']} (paper {b['paper_length']})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=A.REPO / "papers/mrb_morebench/outputs/g2")
    main_from(ap.parse_args().out)
