"""MRB-G1 analysis (MRB_PLAN v0.5 §3-5): tracked judge and generation rows -> aggregates only.

    mrb_g1_analyze.py [--out papers/mrb_morebench/outputs/g1]

Writes ``g1_analysis.json`` beside the rows: per model and field, Regular and Hard for the
paper-faithful generation (k0) on the 500 and on the 400 (DailyDilemmas + AIRiskDilemmas), under
upstream's substring credit (of record) and the exact-token credit (MRB-A3); the registered match
rule (the paper cell inside the 95% prediction interval of a one-generation-per-scenario mean,
resampling our k = 2 generations per scenario, scenarios fixed, widened by the paper's 0.05
rounding); the ladder band; the MDE for a between-model difference; J1 test-retest; the judge
and generation forks; MRB-A10's delimited-prompt discriminator; verdict rates per unit type; and
MB Phase 2's GPT-OSS-20B R2 readout (MB_PLAN Amendments 3-4). Eq. 1 under the registered reading
equals the code score mapped to 0-100, so J0 and J0c share one column (checked, not assumed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent),
                str(REPO / "papers" / "mb_moralbench" / "scripts")]

from mrb_g0_audit import TABLE6, TABLE7  # noqa: E402

from deepsteer.benchmarks.moral_reasoning.morebench.scoring import (  # noqa: E402
    code_score,
    eq1_score,
    eq1_to_paper_scale,
)
from deepsteer.datasets.morebench import load_morebench  # noqa: E402

ANALYSIS_VERSION = "1.0.0"
N_DRAWS, N_BOOT, SEED = 10_000, 2_000, 20261031
ROUNDING = 0.05
PAPER_ROW = {"gpt_oss_20b": "GPT-oss-20b", "qwen3_30b": "Qwen3-30B-A3B"}
PAPER = {"trace": TABLE6, "response": TABLE7}  # Regular at [7], length [8], Hard [9]
SUBSET_400 = ("daily_dilemmas", "ai_risk_dilemmas")
CREDITS = ("credit_substring", "credit_exact")
PARTITIONS = {"daily": ("source", "daily_dilemmas"), "airisk": ("source", "ai_risk_dilemmas"),
              "expert": ("dilemma_type", "expert_case"), "short_case": ("dilemma_type", "short_case"),
              "long_case": ("dilemma_type", "long_case"), "advisor": ("role", "ai_advisor"),
              "agent": ("role", "ai_agent")}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():
        with gzip.open(path.with_suffix(".jsonl.gz"), "rt") as f:
            return [json.loads(x) for x in f if x.strip()]
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def unit_scores(rows: list[dict], weights: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Per judge unit: the code score under each credit test, Eq. 1's check, coverage."""
    by = defaultdict(list)
    for r in rows:
        by[r["unit"]].append(r)
    out = {}
    for unit, rs in by.items():
        sid = rs[0]["unit"].split("/")[2]
        want = weights[sid]
        got = {r["criterion_id"]: r for r in rs}
        rec = {"sid": sid, "complete": set(got) == set(want),
               "no_verdict": sum(r["verdict"] is None for r in rs), "n": len(rs)}
        for c in CREDITS:
            # code_score's input is (weight, judgement); a credited criterion is rendered as the
            # judgement that upstream's test credits, so the score is upstream's arithmetic
            crit = [(w, ("yes" if w > 0 else "no") if got[cid][c] else ("no" if w > 0 else "yes"))
                    for cid, w in want.items() if cid in got]
            rec[c] = code_score(crit)
            if c == "credit_substring":
                rec["eq1_paper_scale"] = eq1_to_paper_scale(eq1_score(crit))
        out[unit] = rec
    return out


def regular(vals: list[float]) -> float:
    return float(np.mean(vals))


def scenario_ci(x: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def match(score: np.ndarray, length: np.ndarray, cell: list, rng) -> dict:
    """``score``/``length``: [n_scenarios, k]. The registered rule on Regular and Hard."""
    n, k = score.shape
    pick = rng.integers(0, k, (N_DRAWS, n))
    rows = np.arange(n)[None, :]
    reg = score[rows, pick].mean(1)
    hard = reg / length[rows, pick].mean(1) * 1000
    out = {}
    for name, dist, paper in (("regular", reg, cell[7]), ("hard", hard, cell[9])):
        lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
        out[name] = {"paper": paper, "pi95": [round(lo, 2), round(hi, 2)],
                     "verdict": ("replicates" if lo - ROUNDING <= paper <= hi + ROUNDING
                                 else "misses"),
                     "paper_minus_pi_mid": round(paper - (lo + hi) / 2, 2)}
    return out


def field_block(us, gens, items, model, field, credit, rng, subset=None) -> dict:
    its = [it for it in items if subset is None or it.source in subset]
    sc = np.array([[us[f"{model}/default/{it.scenario_id}/k{k}/{field}"][credit]
                    for k in (0, 1)] for it in its])
    ln = np.array([[gens[f"{model}/default/{it.scenario_id}/k{k}"][f"{field}_chars"]
                    for k in (0, 1)] for it in its], dtype=float)
    reg0, len0 = regular(sc[:, 0]), float(ln[:, 0].mean())
    cell = PAPER[field][PAPER_ROW[model]]
    out = {"n_scenarios": len(its), "regular_k0": round(reg0, 2),
           "regular_k0_ci95": scenario_ci(sc[:, 0], rng),
           "regular_k_mean": round(float(sc.mean()), 2), "length_k0": round(len0),
           "hard_k0": round(reg0 / len0 * 1000, 2), "paper_regular": cell[7],
           "paper_length": cell[8], "paper_hard": cell[9],
           "scenario_sd_k0": round(float(sc[:, 0].std(ddof=1)), 2),
           "generation_sd_within": round(float(np.sqrt(np.mean(sc.var(1, ddof=1)))), 2)}
    if subset is None:
        out["match"] = match(sc, ln, cell, rng)
        # the paper's partition columns (Daily, AIRisk, Expert, Original, Extended, Advisor,
        # Agent), Original and Extended read as short_case and long_case as in the G0 audit
        parts = {}
        for key, (attr, val) in PARTITIONS.items():
            m = np.array([getattr(it, attr) == val for it in its])
            parts[key] = round(float(sc[m, 0].mean()), 2)
        out["partitions"] = {"ours_k0": parts, "paper": dict(zip(PARTITIONS, cell[:7]))}
    return out


def paired(us, a_units, b_units, credit, rng) -> dict:
    """B - A over the units present in both (scenario-paired), with a scenario-bootstrap CI."""
    keys = [k for k in a_units if k in us and b_units[k] in us]
    d = np.array([us[b_units[k]][credit] - us[k][credit] for k in keys])
    a = np.array([us[k][credit] for k in keys])
    return {"n": len(keys), "regular_a": round(float(a.mean()), 2),
            "regular_b": round(float((a + d).mean()), 2), "diff": round(float(d.mean()), 2),
            "diff_ci95": scenario_ci(d, rng), "diff_sd": round(float(d.std(ddof=1)), 2)}


def criterion_agreement(rows: list[dict], suffix: str) -> dict:
    """Agreement (and Cohen's kappa) of the substring credit between a base unit and its
    ``suffix`` twin (a second pass or another judge) on the same criteria."""
    by = {(r["unit"], r["criterion_id"]): r["credit_substring"] for r in rows}
    pairs = [(v, by[(u + suffix, c)]) for (u, c), v in by.items() if (u + suffix, c) in by]
    if not pairs:
        return {"n": 0}
    x = np.array(pairs, dtype=float)
    po = float((x[:, 0] == x[:, 1]).mean())
    p1, p2 = x[:, 0].mean(), x[:, 1].mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return {"n": len(pairs), "agreement": round(po, 4),
            "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None}


def mb_phase2(root: Path) -> dict | None:
    """MB Phase 2 GPT-OSS-20B R2 at dose-0, both orders: expected score per set (each order
    against its own key), the A-rates, and L and K from letter mass (MB_PLAN Amendment 1)."""
    p = root / "mb_phase2" / "gpt_oss_20b" / "r2.jsonl"
    if not p.exists():
        return None
    import mb_g1_run as mb
    from deepsteer.datasets.moralbench import MORALBENCH_SETS, load_moralbench
    _, items = load_moralbench()
    rows = {(r["set"], r["item_id"], r["order"]): r for r in read_jsonl(p)}
    out = {"meta": json.loads((p.parent / "meta.json").read_text())}
    for s in MORALBENCH_SETS:
        its = [it for it in items if it.set_name == s]
        res = {}
        for order in ("original", "counterbalanced"):
            shown = [mb.counterbalance(it) if order == "counterbalanced" else it for it in its]
            rs = [rows[(s, it.item_id, order)] for it in its]
            res[order] = {
                "expected": round(sum(r["p_A"] * sh.key["A"] + r["p_B"] * sh.key["B"]
                                      for r, sh in zip(rs, shown)), 3),
                "a_rate": round(float(np.mean([r["p_A"] for r in rs])), 4),
                "min_label_mass": round(float(min(r["label_mass"] for r in rs)), 4)}
        a_o, a_c = res["original"]["a_rate"], res["counterbalanced"]["a_rate"]
        res["L"] = round((a_o + a_c) / 2 - 0.5, 4)
        res["K"] = round((a_o - a_c) / 2, 4)
        res["n_items"] = len(its)
        out[s] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "papers/mrb_morebench/outputs/g1")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    items = load_morebench()
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    rows = read_jsonl(a.out / "judge" / "shard0.jsonl")
    gens = {r["job_id"]: r for m in PAPER_ROW for r in read_jsonl(a.out / "gen" / f"{m}.jsonl")}
    us = unit_scores(rows, weights)
    eq1_gap = max(abs(u["eq1_paper_scale"] - u["credit_substring"]) for u in us.values())

    # verdict rates and credit disagreement per unit type
    types = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        p = r["unit"].split("/")
        key = "canned" if p[0] == "canned" else "/".join([p[0], p[4]] + p[5:])
        if p[0] != "canned" and p[1] == "fork":
            key += " (generation fork)"
        t = types[key]
        t[0] += 1
        t[1] += r["verdict"] is None
        t[2] += r["credit_substring"] != r["credit_exact"]
        t[3] += r["n_tokens"]
    verdicts = {k: {"calls": v[0], "no_verdict": v[1], "no_verdict_pct": round(100 * v[1] / v[0], 2),
                    "substring_ne_exact": v[2], "tokens_per_call": round(v[3] / v[0])}
                for k, v in sorted(types.items())}

    fields = {}
    for m in PAPER_ROW:
        for f in ("trace", "response"):
            fields[f"{m}/{f}"] = {
                c: {"all_500": field_block(us, gens, items, m, f, c, rng),
                    "subset_400": field_block(us, gens, items, m, f, c, rng, SUBSET_400)}
                for c in CREDITS}

    # the ladder (100 scenarios) and the models on the same scenarios
    canned = sorted({u.split("/")[1] for u in us if u.startswith("canned/")})
    lad_sids = sorted({us[u]["sid"] for u in us if u.startswith("canned/")})
    ladder = {c: {name: round(regular([us[f"canned/{name}/{s}"][c] for s in lad_sids]), 2)
                  for name in canned} for c in CREDITS}
    on_ladder = {c: {f"{m}/{f}": round(regular([us[f"{m}/default/{s}/k0/{f}"][c]
                                                for s in lad_sids]), 2)
                     for m in PAPER_ROW for f in ("trace", "response")} for c in CREDITS}

    # MDE (80% power, two-sided 5%) for a between-model Regular difference, scenario-paired, k0
    mde = {}
    for f in ("trace", "response"):
        d = np.array([us[f"qwen3_30b/default/{it.scenario_id}/k0/{f}"]["credit_substring"]
                      - us[f"gpt_oss_20b/default/{it.scenario_id}/k0/{f}"]["credit_substring"]
                      for it in items])
        mde[f] = {"n": len(d), "sd_paired_diff": round(float(d.std(ddof=1)), 2),
                  "mde80": round(2.80 * float(d.std(ddof=1)) / np.sqrt(len(d)), 2),
                  "observed_diff_qwen_minus_gptoss": round(float(d.mean()), 2)}

    # J1 retest, the judge fork (top-p 1) and the delimited-prompt fork, on J1's GPT-OSS k0 units
    base = [u for u in us if u.startswith("gpt_oss_20b/default/") and "/k0/" in u
            and len(u.split("/")) == 5 and f"{u}/retest" in us]
    j1 = {}
    for f in ("trace", "response"):
        bf = [u for u in base if u.endswith("/" + f)]
        j1[f] = {
            "retest": {c: paired(us, bf, {u: u + "/retest" for u in bf}, c, rng)
                       for c in CREDITS},
            "judge_fork_top_p_1": {c: paired(us, bf, {u: u + "/judgefork" for u in bf}, c, rng)
                                   for c in CREDITS},
            "retest_criterion_agreement": criterion_agreement(
                [r for r in rows if r["unit"].split("/")[4:5] == [f]], "/retest"),
            "judge_fork_criterion_agreement": criterion_agreement(
                [r for r in rows if r["unit"].split("/")[4:5] == [f]], "/judgefork")}
    bt = [u for u in base if u.endswith("/trace")]
    mrb_a10 = {
        "units": len(bt),
        "no_verdict_registered": sum(us[u]["no_verdict"] for u in bt),
        "no_verdict_delimited": sum(us[u + "/delimfork"]["no_verdict"] for u in bt),
        "calls_each": sum(us[u]["n"] for u in bt),
        "delimited_minus_registered": {c: paired(us, bt, {u: u + "/delimfork" for u in bt}, c,
                                                 rng) for c in CREDITS},
        "delimited_vs_gpt_oss_response": {
            c: paired(us, [u[:-len("trace")] + "response" for u in bt],
                      {u[:-len("trace")] + "response": u + "/delimfork" for u in bt}, c, rng)
            for c in CREDITS}}

    # the generation fork: each model's own defaults, k = 1, on the ladder subset
    gfork = {f"{m}/{f}": {c: paired(us, [f"{m}/default/{s}/k0/{f}" for s in lad_sids],
                                    {f"{m}/default/{s}/k0/{f}": f"{m}/fork/{s}/k0/{f}"
                                     for s in lad_sids}, c, rng) for c in CREDITS}
             for m in PAPER_ROW for f in ("trace", "response")}

    gen_stats = {}
    for m in PAPER_ROW:
        g = [r for k, r in gens.items() if k.startswith(m + "/")]
        gen_stats[m] = {"n": len(g), "final_reached": sum(r["final_reached"] for r in g),
                        "hit_cap": sum(r["finish_reason"] == "length" for r in g),
                        "empty_trace": sum(r["trace_chars"] == 0 for r in g),
                        "trace_chars_median": float(np.median([r["trace_chars"] for r in g])),
                        "response_chars_median": float(np.median([r["response_chars"]
                                                                  for r in g]))}

    out = {"metadata": {"analysis_version": ANALYSIS_VERSION, "date": dt.date.today().isoformat(),
                        "plan": "MRB_PLAN v0.5", "n_draws": N_DRAWS, "n_boot": N_BOOT,
                        "seed": SEED, "rounding": ROUNDING, "judge_rows": len(rows),
                        "units": len(us), "units_incomplete": sum(not u["complete"]
                                                                  for u in us.values()),
                        "eq1_vs_code_max_gap": eq1_gap,
                        "match_rule": "paper cell inside the 95% PI of a one-generation-per-"
                                      "scenario mean (k = 2 resampled, scenarios fixed) +/- 0.05",
                        "credit_of_record": "credit_substring (upstream); credit_exact is MRB-A3"},
           "generation": gen_stats, "verdicts_by_type": verdicts, "fields": fields,
           "ladder": {"scenarios": len(lad_sids), "canned": ladder, "models_same_scenarios":
                      on_ladder}, "mde": mde, "j1": j1, "mrb_a10": mrb_a10,
           "generation_fork": gfork, "mb_phase2": mb_phase2(a.out)}
    (a.out / "g1_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in fields.items():
        b = v["credit_substring"]["all_500"]
        print(f"{k:22s} Regular {b['regular_k0']:6.2f} {b['regular_k0_ci95']} (paper "
              f"{b['paper_regular']}) {b['match']['regular']['verdict']:10s} PI "
              f"{b['match']['regular']['pi95']} | Hard {b['hard_k0']:6.2f} (paper "
              f"{b['paper_hard']}) {b['match']['hard']['verdict']} | len {b['length_k0']} "
              f"(paper {b['paper_length']})")


if __name__ == "__main__":
    main()
