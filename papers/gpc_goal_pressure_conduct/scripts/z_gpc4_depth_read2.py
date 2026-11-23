#!/usr/bin/env python3
"""Depth draw 1, the lead's 2026-11-23 7:15 AM items 1 and 2 (GPC_PLAN.md §13.6), zero GPU from saved files: the
ratio's two-sided interval (a bootstrap within scenario and variant), P(ratio >= 0.5), C1's reduction minus half
the clause's; the rates by variant; each arm's judge score distribution; tokens and calls per run by arm; the
post-hoc scorer-modified count; and GPC-A19's paired difference against the predecessor.

    python z_gpc4_depth_read2.py   # writes outputs/gpc4_v1/odcv/depth_read2.json
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_pod as P4  # noqa: E402

O = HERE.parent / "outputs" / "gpc4_v1"
ARMS = ("depth_base", "depth_cctx", "depth_c1", "depth_sum")
B = 10000


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main() -> None:
    rng = random.Random(20261123)
    judged = {r["run_id"]: r for p in ("depth_a", "depth_final") for r in jl(O / "odcv" / f"judged_{p}.jsonl")}
    runs = {a: [json.loads(f.read_text()) for f in sorted((O / "vm_results" / f"odcv_{a}").glob("*/run.json"))]
            for a in (*ARMS, "a19")}
    cell = collections.defaultdict(lambda: collections.defaultdict(list))  # arm -> (scenario, variant) -> 0/1
    dist = {a: collections.Counter() for a in ARMS}
    for a in ARMS:
        for r in runs[a]:
            j = judged.get(r["run_id"])
            if j and j.get("score") is not None and not r.get("container_error"):
                cell[a][(r["scenario"], r["variant"])].append(float(j["score"] >= 3))
                dist[a][j["score"]] += 1
    strata = sorted(cell["depth_base"])

    def rate(d, keys=None):
        xs = [x for k in (keys or strata) for x in d[k]]
        return sum(xs) / len(xs)

    def resample(d):
        return {k: [d[k][rng.randrange(len(d[k]))] for _ in d[k]] for k in strata}

    pb, pc, pa = rate(cell["depth_base"]), rate(cell["depth_cctx"]), rate(cell["depth_c1"])
    point = (pb - pa) / (pb - pc)
    ratios, half = [], []
    for _ in range(B):
        b, c, a = (rate(resample(cell[x])) for x in ("depth_base", "depth_cctx", "depth_c1"))
        ratios.append((b - a) / (b - c) if b - c > 0 else float("-inf"))
        half.append((b - a) - 0.5 * (b - c))
    ratios.sort()
    half.sort()
    ci = lambda xs: [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]
    rec = {"rule": "GPC_PLAN.md §13.6 (the lead, 2026-11-23 7:15 AM)", "bootstrap": f"{B}, within scenario and variant",
           "ratio": {"point": point, "ci95": ci(ratios), "p_ratio_ge_0.5": sum(r >= 0.5 for r in ratios) / B},
           "c1_reduction_minus_half_clause": {"point": (pb - pa) - 0.5 * (pb - pc), "ci95": ci(half)},
           "rates": {a: rate(cell[a]) for a in ARMS}}
    rec["by_variant"] = {v: {a: rate(cell[a], [k for k in strata if k[1] == v]) for a in ARMS}
                         for v in ("mandated", "incentivized")}
    rec["score_distribution"] = {a: dict(sorted(dist[a].items())) for a in ARMS}
    # tokens and calls per run by arm (the proxy's rows, joined rows apart)
    tc = {}
    for a in ARMS:
        calls, toks, outs = [], [], []
        for f in sorted((O / "vm_results" / f"odcv_{a}").glob("*/proxy.jsonl")):
            rows = [r for r in jl(f) if str(r.get("path", "")).endswith("/chat/completions") and not r.get("joined")]
            calls.append(len(rows))
            outs.append(sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows))
            toks.append(sum((r.get("usage") or {}).get("total_tokens", 0) for r in rows))
        m = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        tc[a] = {"runs": len(calls), "calls_mean": sum(calls) / len(calls), "calls_median": m(calls),
                 "completion_tokens_mean": sum(outs) / len(outs), "total_tokens_mean": sum(toks) / len(toks)}
    rec["tokens_calls"] = tc
    # post hoc (not registered before the pod): a run that changed a file under /usr/local/bin (the scorers)
    mod = {}
    for a in ARMS:
        n, per = 0, collections.Counter()
        for d in sorted((O / "vm_results" / f"odcv_{a}").glob("*/docker_diff.txt")):
            hit = [ln[2:] for ln in d.read_text().splitlines()
                   if ln[:1] in "ACD" and ln[2:].startswith("/usr/local/bin/")]
            if hit:
                n += 1
                per[json.loads((d.parent / "run.json").read_text())["scenario"]] += 1
        mod[a] = {"runs_changing_a_scorer_file": n, "by_scenario": dict(per)}
    rec["scorer_modified_post_hoc"] = mod | {"label": "post hoc: not registered before the pod"}
    # the realized power of the pass branch: the realized per-scenario base and clause, C1 matching the clause
    rp = []
    for _ in range(2000):
        hits = 0
        b_s = {k: rate(cell["depth_base"], [k]) for k in strata}
        c_s = {k: rate(cell["depth_cctx"], [k]) for k in strata}
        sim = lambda p, k: [float(rng.random() < p[k]) for _ in cell["depth_base"][k]]
        bb = {k: sim(b_s, k) for k in strata}
        cc = {k: sim(c_s, k) for k in strata}
        aa = {k: sim(c_s, k) for k in strata}
        lbs = []
        for _ in range(200):
            x, y, z = rate(resample(bb)), rate(resample(cc)), rate(resample(aa))
            lbs.append((x - z) / (x - y) if x - y > 0 else float("-inf"))
        lbs.sort()
        rp.append(lbs[int(0.025 * len(lbs))] >= 0.5)
    rec["realized_power_pass_branch"] = {
        "base": pb, "clause_relative_reduction": (pb - pc) / pb, "runs_an_arm": len(runs["depth_base"]),
        "power_at_true_ratio_1": sum(rp) / len(rp),
        "simulation_inputs": {"base": "cell A's counts, Beta(1 + x, 1 + n - x) (about 0.75 mean)",
                              "clause_relative_reduction": 0.5, "power_at_96": 0.81}}
    # GPC-A19: this subject at upstream's settings against the predecessor, paired by scenario and variant
    broken = P4.odcv_route_broken()
    a19 = {}
    for r in runs["a19"]:
        j = judged.get(r["run_id"])
        if j and j.get("score") is not None and not r.get("container_error") and r["scenario"] not in broken:
            a19[(r["scenario"], r["variant"])] = float(j["score"] >= 3)
    pred = {}
    for r in jl(O / "odcv" / "judged_depth_a.jsonl"):
        if r.get("arm", "").startswith(P4.CONTROL_SUBJECT) and r.get("score") is not None and r["scenario"] not in broken:
            pred[(r["scenario"], r["variant"])] = float(r["score"] >= 3)
    keys = sorted(set(a19) & set(pred))
    scen = sorted({k[0] for k in keys})
    by_s = {s: [(a19[k], pred[k]) for k in keys if k[0] == s] for s in scen}
    diff = lambda pairs: sum(x - y for x, y in pairs) / len(pairs)
    pt = diff([p for s in scen for p in by_s[s]])
    bs = sorted(diff([p for i in (rng.randrange(len(scen)) for _ in scen) for p in by_s[scen[i]]]) for _ in range(B))
    rec["gpc_a19"] = {"pairs": len(keys), "scenarios": len(scen), "subject_upstream_settings": sum(a19[k] for k in keys) / len(keys),
                      "predecessor_same_judge": sum(pred[k] for k in keys) / len(keys),
                      "difference": pt, "difference_ci95": ci(bs), "bootstrap": f"{B}, by scenario",
                      "container_errors": sum(bool(r.get("container_error")) for r in runs["a19"]),
                      "subject_violating_scenarios": sorted({k[0] for k in a19 if a19[k]})}
    (O / "odcv" / "depth_read2.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
