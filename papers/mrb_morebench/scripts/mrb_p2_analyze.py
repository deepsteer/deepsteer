"""MRB Phase 2 analysis (MRB_PLAN v0.9; MB_PLAN Amendment 5) -> ``p2_analysis.json``.

    mrb_p2_analyze.py [--out papers/mrb_morebench/outputs/p2]

Per model: the headline Δ (adapter − none, each scenario's mean over its generations, the code score
of record) with a scenario-bootstrap 95% CI; the five dimensions (share of criteria satisfied,
pooled within a scenario; raw and Holm-adjusted); the 400-scenario subset; the length rule (Δ length
CI excludes 0 → ΔHard beside ΔRegular, and a Regular gain without a same-direction Hard shift whose
CI excludes 0 is "length-confounded"); the registered branch; MDEs from the measured spread. The
secondary ΔΔ (Muse − Qwen) on Harmless Outcome, Logical Process and Regular, jointly resampled. The
MoralBench rider: expected score per set in each order against its own key, A-rates, L and K, and
the adapter's shifts with item-bootstrap CIs; the merge and parity records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "mb_moralbench" / "scripts")]

import mrb_g1_analyze as A  # noqa: E402

ANALYSIS_VERSION = "1.0.0"
MODELS = ("muse_glimmer_30b", "qwen38_27b")
DIMS = ("identifying", "logical process", "clear process", "helpful outcome", "harmless outcome")
SECONDARY = ("harmless outcome", "logical process")
SUBSET_400 = ("daily_dilemmas", "ai_risk_dilemmas")
N_BOOT, SEED = 2000, 20261101


def ci(x: np.ndarray) -> list[float]:
    return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]


def boot_p(x: np.ndarray) -> float:
    """Two-sided bootstrap p for a mean of 0 (percentile method)."""
    return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = round(run, 4)
    return out


def excludes_zero(c: list[float]) -> bool:
    return c[0] > 0 or c[1] < 0


def branch(delta_ci: list[float], mde: float, confounded: bool) -> str:
    """The registered branch for one model's headline (MRB_PLAN v0.9)."""
    if delta_ci[0] > 0:
        return "length-confounded gain" if confounded else "raises process quality"
    if delta_ci[1] < 0:
        return "process-quality cost"
    return f"no change detectable at |Δ| ≳ {mde:.2f}"


def per_scenario(us, dim_rates, gens, model, arm, sids):
    """Per scenario: mean over k of the score, of each dimension's rate, and of response length."""
    ks = sorted({int(g.split("/")[3][1:]) for g in gens if g.startswith(f"{model}/{arm}/")})
    sc, ln, dims = [], [], {d: [] for d in DIMS}
    for s in sids:
        units = [f"{model}/{arm}/{s}/k{k}/response" for k in ks]
        sc.append(np.mean([us[u]["credit_substring"] for u in units]))
        ln.append(np.mean([gens[u.rsplit("/", 1)[0]]["response_chars"] for u in units]))
        for d in DIMS:
            v = [dim_rates[u][d] for u in units if d in dim_rates[u]]
            dims[d].append(np.mean(v) if v else np.nan)
    return np.array(sc), np.array(ln), {d: np.array(v) for d, v in dims.items()}, ks


def model_block(us, dim_rates, gens, model, items, rng) -> dict:
    sids = [it.scenario_id for it in items]
    s0, l0, d0, ks = per_scenario(us, dim_rates, gens, model, "none", sids)
    s1, l1, d1, _ = per_scenario(us, dim_rates, gens, model, "adapter", sids)
    n = len(sids)
    idx = rng.integers(0, n, (N_BOOT, n))
    delta = s1 - s0
    bd = delta[idx].mean(1)
    dl = (l1 - l0)[idx].mean(1)
    hard = (s1[idx].mean(1) / l1[idx].mean(1) - s0[idx].mean(1) / l0[idx].mean(1)) * 1000
    d_ci, l_ci, h_ci = ci(bd), ci(dl), ci(hard)
    length_rule = excludes_zero(l_ci)
    confounded = bool(length_rule and d_ci[0] > 0 and not h_ci[0] > 0)
    mde = 2.8 * float(delta.std(ddof=1)) / np.sqrt(n)
    dims, ps = {}, {}
    for d in DIMS:
        x = d1[d] - d0[d]
        ok = ~np.isnan(x)
        xb = np.nanmean(np.where(ok[idx], x[idx], np.nan), axis=1)
        ps[d] = boot_p(xb)
        dims[d] = {"n": int(ok.sum()), "none": round(float(np.nanmean(d0[d])), 2),
                   "adapter": round(float(np.nanmean(d1[d])), 2),
                   "delta": round(float(np.nanmean(x)), 3), "ci95": ci(xb), "p_boot": ps[d],
                   "mde80": round(2.8 * float(np.nanstd(x, ddof=1)) / np.sqrt(ok.sum()), 2)}
    for d, p in holm(ps).items():
        dims[d]["p_holm"] = p
    sub = np.array([it.source in SUBSET_400 for it in items])
    bsub = delta[sub][rng.integers(0, sub.sum(), (N_BOOT, sub.sum()))].mean(1)
    within = None
    if len(ks) >= 2:  # the generation noise of a single arm, k0 against k1
        g = [us[f"{model}/none/{s}/k0/response"]["credit_substring"]
             - us[f"{model}/none/{s}/k1/response"]["credit_substring"] for s in sids]
        within = round(float(np.std(g, ddof=1) / np.sqrt(2)), 2)
    out = {"k": len(ks), "n": n, "regular_none": round(float(s0.mean()), 2),
           "regular_adapter": round(float(s1.mean()), 2), "delta": round(float(delta.mean()), 3),
           "delta_ci95": d_ci, "mde80_observed": round(mde, 2),
           "per_generation_sd_none": within,
           "mde80_uniform_effect": (round(2.8 * within * np.sqrt(2 / len(ks)) / np.sqrt(n), 2)
                                    if within else None),
           "length_none": round(float(l0.mean())), "length_adapter": round(float(l1.mean())),
           "delta_length": round(float((l1 - l0).mean()), 1), "delta_length_ci95": l_ci,
           "length_rule_applies": length_rule, "subset_400": {"n": int(sub.sum()),
                                                             "delta": round(float(delta[sub].mean()), 3),
                                                             "ci95": ci(bsub)},
           "dimensions": dims}
    if length_rule:
        out["delta_hard"] = round(float((s1.mean() / l1.mean() - s0.mean() / l0.mean()) * 1000), 3)
        out["delta_hard_ci95"] = h_ci
    out["length_confounded"] = confounded
    out["branch"] = branch(d_ci, mde, confounded)
    out["_arrays"] = {"delta": delta, "dims": {d: d1[d] - d0[d] for d in DIMS}}
    return out


def secondary(blocks: dict, n: int, rng) -> dict:
    """ΔΔ = Δ_Muse − Δ_Qwen per scenario, jointly resampled (the same 500 scenarios)."""
    a, b = blocks["muse_glimmer_30b"]["_arrays"], blocks["qwen38_27b"]["_arrays"]
    idx = rng.integers(0, n, (N_BOOT, n))
    out = {}
    for name, x in [("regular", a["delta"] - b["delta"])] + \
            [(d, a["dims"][d] - b["dims"][d]) for d in SECONDARY]:
        ok = ~np.isnan(x)
        xb = np.nanmean(np.where(ok[idx], x[idx], np.nan), axis=1)
        c = ci(xb)
        mde = 2.8 * float(np.nanstd(x, ddof=1)) / np.sqrt(ok.sum())
        out[name] = {"delta_delta": round(float(np.nanmean(x)), 3), "ci95": c,
                     "mde80": round(mde, 2),
                     "reading": ("separates the two adapters" if excludes_zero(c) else
                                 f"no separation detectable at |ΔΔ| ≳ {mde:.2f} (not a verdict "
                                 "against MoReBench: ST's non-specificity was measured on "
                                 "rule-right items)")}
    return out


def mb_rider(out: Path, rng) -> dict:
    import mb_g1_run as mb

    from deepsteer.datasets.moralbench import MORALBENCH_SETS, load_moralbench
    _, items = load_moralbench()
    res = {}
    for m in MODELS:
        d = out / "mb_phase2" / m
        if not (d / "r2_none.jsonl").exists():
            continue
        arms = {arm: {(r["set"], r["item_id"], r["order"]): r for r in A.read_jsonl(
            d / f"r2_{arm}.jsonl")} for arm in ("none", "adapter") if (d / f"r2_{arm}.jsonl").exists()}
        per = {"hf_phase": json.loads((d / "hf_phase.json").read_text())}
        for s in MORALBENCH_SETS:
            its = [it for it in items if it.set_name == s]
            cb = {it.item_id: mb.counterbalance(it) for it in its}

            def stats(rows, sel):
                g = [its[i] for i in sel]
                eo = sum(rows[(s, it.item_id, "original")]["p_A"] * it.key["A"]
                         + rows[(s, it.item_id, "original")]["p_B"] * it.key["B"] for it in g)
                ec = sum(rows[(s, it.item_id, "counterbalanced")]["p_A"] * cb[it.item_id].key["A"]
                         + rows[(s, it.item_id, "counterbalanced")]["p_B"]
                         * cb[it.item_id].key["B"] for it in g)
                ao = float(np.mean([rows[(s, it.item_id, "original")]["p_A"] for it in g]))
                ac = float(np.mean([rows[(s, it.item_id, "counterbalanced")]["p_A"] for it in g]))
                return {"expected_original": eo, "expected_counterbalanced": ec, "a_o": ao,
                        "a_c": ac, "L": (ao + ac) / 2 - 0.5, "K": (ao - ac) / 2}

            full = list(range(len(its)))
            row = {arm: {k: round(v, 4) for k, v in stats(r, full).items()}
                   for arm, r in arms.items()}
            if "adapter" in arms:
                bs = [rng.integers(0, len(its), len(its)) for _ in range(N_BOOT)]
                for key in ("expected_original", "expected_counterbalanced", "L", "K"):
                    diffs = np.array([stats(arms["adapter"], b)[key] - stats(arms["none"], b)[key]
                                      for b in bs])
                    row[f"delta_{key}"] = {"delta": round(row["adapter"][key] - row["none"][key], 4),
                                           "ci95": ci(diffs)}
            row["n_items"] = len(its)
            per[s] = row
        res[m] = per
    return res


def main_from(out: Path) -> dict:
    from deepsteer.datasets.morebench import load_morebench
    rng = np.random.default_rng(SEED)
    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    dim_of = {(it.scenario_id, c.criterion_id): c.dimension for it in items for c in it.criteria}
    rows = A.read_jsonl(out / "judge" / "shard0.jsonl")
    us = A.unit_scores(rows, weights)
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        d = dim_of[(r["unit"].split("/")[2], r["criterion_id"])]
        acc[r["unit"]][d][0] += r["credit_substring"]
        acc[r["unit"]][d][1] += 1
    dim_rates = {u: {d: 100 * v[0] / v[1] for d, v in ds.items()} for u, ds in acc.items()}
    gens = {}
    for f in sorted((out / "gen").glob("*.jsonl")):
        gens.update({r["job_id"]: r for r in A.read_jsonl(f)})
    have = {u.rsplit("/", 1)[0] for u in us}
    items = [it for it in items if all(f"{m}/{arm}/{it.scenario_id}/k0" in have
                                       for m in MODELS for arm in ("none", "adapter")
                                       if any(g.startswith(f"{m}/{arm}/") for g in gens))]
    blocks = {m: model_block(us, dim_rates, gens, m, items, rng) for m in MODELS
              if any(g.startswith(f"{m}/adapter/") for g in gens)}
    sec = secondary(blocks, len(items), rng) if len(blocks) == 2 else None
    verdicts = defaultdict(lambda: [0, 0])
    for r in rows:
        key = "/".join(r["unit"].split("/")[:2])
        verdicts[key][0] += 1
        verdicts[key][1] += r["verdict"] is None
    for b in blocks.values():
        b.pop("_arrays")
    res = {"metadata": {"analysis_version": ANALYSIS_VERSION, "date": dt.date.today().isoformat(),
                        "plan": "MRB_PLAN v0.9", "n_boot": N_BOOT, "seed": SEED,
                        "judge_rows": len(rows), "units_incomplete": sum(not u["complete"]
                                                                         for u in us.values()),
                        "scenarios": len(items)},
           "headline": blocks, "secondary": sec,
           "verdicts_by_arm": {k: {"calls": v[0], "no_verdict": v[1]} for k, v in verdicts.items()},
           "generation": {k.removesuffix(".summary.json"): json.loads(p.read_text())
                          for p in sorted((out / "gen").glob("*.summary.json"))
                          for k in [p.name]},
           "mb_rider": mb_rider(out, rng)}
    (out / "p2_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for m, b in blocks.items():
        print(f"{m}: Δ {b['delta']:+.2f} {b['delta_ci95']} (k={b['k']}, MDE {b['mde80_observed']})"
              f" | Δlen {b['delta_length']:+.0f} {b['delta_length_ci95']} | {b['branch']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=A.REPO / "papers/mrb_morebench/outputs/p2")
    main_from(ap.parse_args().out)
