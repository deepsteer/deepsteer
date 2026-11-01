"""R1-R4 of the MRB-G1 OpenRouter cross-check (MRB_PLAN v0.6 §5) -> ``g1_openrouter_analysis.json``.

    mrb_g1_openrouter_analyze.py [--out papers/mrb_morebench/outputs/g1_openrouter]

R1: X2 (GPT-oss-20b through OpenRouter, the paper's settings) per field: lengths, Regular and Hard
on J1's 50 scenarios with scenario-bootstrap CIs, read against Tables 6 and 7 and against our pod
on the same scenarios. R2: the OpenRouter judge on our pod's k0 generations against the pod judge
(GPT-OSS: the pod's top-p 1 pass; Qwen: the pod's default). R3: the response cells' sensitivity
interval (the registered PI +/- 0.05, widened by R2's larger absolute CI end), and both
near-misses against the 1.96-point MDE. R4: X2 and X2b trace lengths against the pod's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mrb_g1_analyze as A  # noqa: E402
import mrb_g1_openrouter as O  # noqa: E402

MDE = 1.96  # the G1 analysis's between-model MDE on responses (g1_analysis.json, mde.response)


def _scores(us: dict, units: list[str], credit: str) -> np.ndarray:
    return np.array([us[u][credit] for u in units])


def main_from(out: Path) -> dict:
    rng = np.random.default_rng(A.SEED)
    items = O.scenarios()
    sids = [it.scenario_id for it in items]
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    rows = A.read_jsonl(out / "judge.jsonl")
    us = A.unit_scores(rows, weights)
    pod_rows = [r for r in A.read_jsonl(O.POD_OUT / "judge" / "shard0.jsonl")
                if r["unit"].split("/")[2:3] and r["unit"].split("/")[2] in weights]
    pod = A.unit_scores(pod_rows, weights)
    pod_an = json.loads((O.POD_OUT / "g1_analysis.json").read_text())
    gens = {r["job_id"]: r for r in A.read_jsonl(out / "gen.jsonl")}
    pod_gens = {r["job_id"]: r for m in A.PAPER_ROW
                for r in A.read_jsonl(O.POD_OUT / "gen" / f"{m}.jsonl")}

    r1 = {}
    for f in ("trace", "response"):
        units = [f"gpt_oss_20b_or/paper/{s}/k0/{f}" for s in sids]
        have = [u for u in units if u in us and us[u]["complete"]]
        cell = A.PAPER[f]["GPT-oss-20b"]
        ln = np.array([gens[u.rsplit("/", 1)[0]][f"{f}_chars"] for u in have], dtype=float)
        rec = {"n": len(have), "length_mean": round(float(ln.mean()), 1) if len(ln) else None,
               "paper_length": cell[8], "paper_regular": cell[7], "paper_hard": cell[9],
               "no_verdict": sum(us[u]["no_verdict"] for u in have),
               "calls": sum(us[u]["n"] for u in have)}
        for c in A.CREDITS:
            x = _scores(us, have, c)
            reg, ci = float(x.mean()), A.scenario_ci(x, rng)
            rec[c] = {"regular": round(reg, 2), "regular_ci95": ci,
                      "hard": round(reg / ln.mean() * 1000, 2),
                      "paper_regular_in_ci": ci[0] - A.ROUNDING <= cell[7] <= ci[1] + A.ROUNDING}
        pod_units = [f"gpt_oss_20b/default/{s}/k0/{f}" for s in sids]
        rec["pod_same_scenarios"] = {
            "length_mean": round(float(np.mean([pod_gens[u.rsplit("/", 1)[0]][f"{f}_chars"]
                                                for u in pod_units])), 1),
            "regular_registered_prompt": round(float(_scores(pod, pod_units,
                                                             "credit_substring").mean()), 2)}
        if f == "trace":
            rec["pod_same_scenarios"]["regular_delimited_prompt"] = round(float(np.mean(
                [pod[u + "/delimfork"]["credit_substring"] for u in pod_units])), 2)
        r1[f] = rec

    r2 = {}
    for m in A.PAPER_ROW:
        for f in ("trace", "response"):
            base = [f"{m}/default/{s}/k0/{f}" for s in sids]
            twin = {u: u + "/judgefork" if m == "gpt_oss_20b" else u for u in base}
            keys = [u for u in base if u in us and us[u]["complete"]]
            rec = {"pod_reference": "pod top-p 1 (J1 judge fork)" if m == "gpt_oss_20b"
                   else "pod default (top-p 0.95)", "n": len(keys),
                   "no_verdict_openrouter": sum(us[u]["no_verdict"] for u in keys),
                   "no_verdict_pod": sum(pod[twin[u]]["no_verdict"] for u in keys),
                   "calls": sum(us[u]["n"] for u in keys)}
            for c in A.CREDITS:
                d = np.array([us[u][c] - pod[twin[u]][c] for u in keys])
                rec[c] = {"regular_openrouter": round(float(_scores(us, keys, c).mean()), 2),
                          "regular_pod": round(float(np.mean([pod[twin[u]][c] for u in keys])), 2),
                          "diff": round(float(d.mean()), 2), "diff_ci95": A.scenario_ci(d, rng)}
            by_pod = {(r["unit"], r["criterion_id"]): r["credit_substring"] for r in pod_rows}
            agree = [(r["credit_substring"], by_pod[(twin[r["unit"]], r["criterion_id"])])
                     for r in rows if r["unit"] in twin
                     and (twin[r["unit"]], r["criterion_id"]) in by_pod]
            x = np.array(agree, dtype=float)
            rec["criterion_agreement"] = round(float((x[:, 0] == x[:, 1]).mean()), 4) \
                if len(x) else None
            r2[f"{m}/{f}"] = rec

    r3 = {}
    for m in A.PAPER_ROW:
        blk = pod_an["fields"][f"{m}/response"]["credit_substring"]["all_500"]
        lo, hi = blk["match"]["regular"]["pi95"]
        paper = blk["match"]["regular"]["paper"]
        ci = r2[f"{m}/response"]["credit_substring"]["diff_ci95"]
        delta = max(abs(ci[0]), abs(ci[1]))
        wlo, whi = lo - A.ROUNDING - delta, hi + A.ROUNDING + delta
        outside = max(lo - A.ROUNDING - paper, paper - hi - A.ROUNDING, 0.0)
        r3[m] = {"registered_pi95": [lo, hi], "registered_verdict": blk["match"]["regular"]
                 ["verdict"], "paper": paper, "ours_k0": blk["regular_k0"], "delta": delta,
                 "sensitivity_interval": [round(wlo, 2), round(whi, 2)],
                 "sensitivity_verdict": ("replicates within deployment noise"
                                         if wlo <= paper <= whi else "misses"),
                 "outside_registered_by": round(outside, 2),
                 "outside_as_share_of_mde": round(outside / MDE, 3),
                 "paper_minus_ours": round(paper - blk["regular_k0"], 2),
                 "paper_minus_ours_as_share_of_mde": round((paper - blk["regular_k0"]) / MDE, 3)}

    r4 = {}
    for arm in O.GEN_ARMS:
        g = [r for k, r in gens.items() if k.split("/")[1] == arm]
        r4[arm] = {"n": len(g),
                   "trace_chars_mean": round(float(np.mean([r["trace_chars"] for r in g])), 1)
                   if g else None,
                   "response_chars_mean": round(float(np.mean([r["response_chars"] for r in g])),
                                                1) if g else None,
                   "reasoning_tokens_mean": (round(float(np.mean([r["reasoning_tokens"] for r in g
                                                                  if r["reasoning_tokens"]])), 1)
                                             if any(r["reasoning_tokens"] for r in g) else None),
                   "empty_trace": sum(r["trace_chars"] == 0 for r in g),
                   "finish_length": sum(r["finish_reason"] == "length" for r in g),
                   "providers": dict(Counter(r["provider"] for r in g))}
    r4["pod_effort_high"] = {
        "trace_chars_mean": round(float(np.mean([pod_gens[f"gpt_oss_20b/default/{s}/k0"]
                                                 ["trace_chars"] for s in sids])), 1),
        "response_chars_mean": round(float(np.mean([pod_gens[f"gpt_oss_20b/default/{s}/k0"]
                                                    ["response_chars"] for s in sids])), 1)}
    spent = {k: sum(int(r.get(k) or 0) for r in list(gens.values()) + rows)
             for k in ("prompt_tokens", "completion_tokens")}
    res = {"metadata": {"plan": "MRB_PLAN v0.6", "scenarios": len(sids), "judge_rows": len(rows),
                        "tokens": spent, "judge_providers": dict(Counter(r["provider"]
                                                                         for r in rows))},
           "r1_paper_row": r1, "r2_deployment": r2, "r3_near_misses": r3, "r4_lengths": r4}
    (out / "g1_openrouter_analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"r1": {f: {k: r1[f][k] for k in ("n", "length_mean", "no_verdict")}
                             | {"regular": r1[f]["credit_substring"]["regular"]}
                             for f in r1}, "r4": r4}, indent=None))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=O.DEFAULT_OUT)
    main_from(ap.parse_args().out)
