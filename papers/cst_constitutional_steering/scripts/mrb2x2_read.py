#!/usr/bin/env python3
"""The MoReBench pod's registered read (CST Amendment 46 item B4-B7), zero GPU.

    /opt/homebrew/bin/python3.13 mrb2x2_read.py DIR

Each arm (A0s, P-ctx, P-ctx-n; config 2's saved read beside) against MRB's saved base rows at
matched length (MRB's v0.9.3 estimator, ``mrb_p2_length``: beta from the base's within-scenario
coupling, scenario bootstrap, MRB's seed), each contrast's MDE (2.80 x its bootstrap SE), and the
contrasts between arms as differences of their matched-length costs in the same scenario resamples.
Readings: **frame** (A0s's CI excludes 0); **generic tuning** (A0s includes 0 and P-ctx-n excludes
0); **distilling this document** (A0s and P-ctx-n include 0 and P-ctx minus P-ctx-n excludes 0);
else **not separable at this design**. The top-up rule (A0s's CI includes both 0 and -0.93), the
length rule (cap-hit shares; past the base's by 0.05, a read on scenarios where neither arm capped
beside) and the secondary fingerprint (the frozen consequence criteria, raw score difference) are
applied.
Writes ``DIR/mrb2x2_read.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "mrb_morebench/scripts")]

CST = HERE.parent
MODEL = "qwen38_27b"
MRB_P2 = HERE.parents[1] / "mrb_morebench/outputs/p2"
BARS = CST / "outputs/finalist_c2_rule_right_bars/mrb"
FINGERPRINT = CST / "data/morebench_consequence_criteria_v1.json"
ARMS = {"a0s": None, "p_ctx": None, "p_ctx_n": None, "c2_rule_right": BARS}
MRB05 = -0.93


def _units(rows, arm_from: str, arm_to: str):
    return [dict(r, unit=r["unit"].replace(f"{MODEL}/{arm_from}/", f"{MODEL}/{arm_to}/"))
            for r in rows]


def load(out: Path):
    import mrb_g1_analyze as A

    from deepsteer.datasets.morebench import load_morebench

    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    judge = {"none": [r for r in A.read_jsonl(MRB_P2 / "judge" / "shard0.jsonl")
                      if r["unit"].startswith(f"{MODEL}/none/")]}
    gens = {"none": {r["job_id"]: r for r in A.read_jsonl(MRB_P2 / "gen" / f"{MODEL}.none.jsonl")}}
    own = out / "mrb" / "judge" / "shard0.jsonl"
    own_rows = A.read_jsonl(own) if own.exists() else []
    for arm, src in ARMS.items():
        if src is None:
            g = out / "mrb" / "gen" / f"{MODEL}.{arm}.jsonl"
            if g.exists():
                judge[arm] = [r for r in own_rows if r["unit"].startswith(f"{MODEL}/{arm}/")]
                gens[arm] = {r["job_id"]: r for r in A.read_jsonl(g)}
        elif (src / "judge" / "shard0.jsonl").exists():
            judge[arm] = _units(A.read_jsonl(src / "judge" / "shard0.jsonl"), "candidate", arm)
            gens[arm] = {r["job_id"].replace("/candidate/", f"/{arm}/"): r for r in
                         A.read_jsonl(src / "gen" / f"{MODEL}.candidate.jsonl")}
    return items, weights, judge, gens


def arrays(us, gens, arm, sids):
    sc = np.array([[us[f"{MODEL}/{arm}/{s}/k{k}/response"]["credit_substring"] for k in (0, 1)]
                   for s in sids])
    ln = np.array([[gens[arm][f"{MODEL}/{arm}/{s}/k{k}"]["response_chars"] for k in (0, 1)]
                   for s in sids], dtype=float) / 1000.0
    return sc, ln


def adj(base, arm, idx=None):
    """MRB v0.9.3: delta less beta x delta-length, beta from the base's k1 - k0 coupling."""
    (s0, l0), (s1, l1) = base, arm
    if idx is not None:
        s0, l0, s1, l1 = s0[idx], l0[idx], s1[idx], l1[idx]
    ds, dl = s0[:, 1] - s0[:, 0], l0[:, 1] - l0[:, 0]
    beta = float((ds * dl).sum() / (dl * dl).sum())
    return float((s1.mean(1) - s0.mean(1)).mean() - beta * (l1.mean(1) - l0.mean(1)).mean())


def complete(us, gens, arm, items) -> list[str]:
    return [it.scenario_id for it in items
            if all(us.get(f"{MODEL}/{a}/{it.scenario_id}/k{k}/response", {}).get("complete")
                   and f"{MODEL}/{a}/{it.scenario_id}/k{k}" in gens.get(a, {})
                   for a in ("none", arm) for k in (0, 1))]


def read(out: Path) -> dict:
    import mrb_g1_analyze as A
    import mrb_p2_length as LEN

    items, weights, judge, gens = load(out)
    us = {}
    for rows in judge.values():
        us |= A.unit_scores(rows, weights)
    arms = [a for a in ARMS if a in judge and judge[a]]
    common = sorted(set.intersection(*[set(complete(us, gens, a, items)) for a in arms]))
    if len(common) < 2:
        return {"rule": "Amendment 46 item B", "scenarios_read": len(common)}
    base = arrays(us, gens, "none", common)
    arr = {a: arrays(us, gens, a, common) for a in arms}
    rng = np.random.default_rng(LEN.SEED)
    boots = [rng.integers(0, len(common), len(common)) for _ in range(LEN.N_BOOT)]
    draws = {a: np.array([adj(base, arr[a], b) for b in boots]) for a in arms}
    point = {a: adj(base, arr[a]) for a in arms}

    def summ(p, d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"est": round(p, 4), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "mde": round(2.80 * float(np.std(d)), 3), "excludes_0": bool(lo > 0 or hi < 0)}

    res = {"rule": "Amendment 46 item B (matched length, paired by scenario, against MRB's saved "
                   "base)", "scenarios_read": len(common), "arms_read": arms,
           "vs_base": {a: summ(point[a], draws[a]) for a in arms}, "contrasts": {}}
    for x, y in (("p_ctx", "p_ctx_n"), ("a0s", "c2_rule_right"), ("a0s", "p_ctx")):
        if x in arms and y in arms:
            res["contrasts"][f"{x}_minus_{y}"] = summ(point[x] - point[y], draws[x] - draws[y])
    v, c = res["vs_base"], res["contrasts"]
    a0s_x = v.get("a0s", {}).get("excludes_0")
    n_x = v.get("p_ctx_n", {}).get("excludes_0")
    d_x = c.get("p_ctx_minus_p_ctx_n", {}).get("excludes_0")
    if a0s_x:
        reading = "frame (A0n at k 2 follows)"
    elif a0s_x is False and n_x:
        reading = "generic tuning"
    elif a0s_x is False and n_x is False and d_x:
        reading = "distilling this document"
    else:
        reading = "not separable at this design"
    res["reading"] = reading
    if "a0s" in v:
        lo, hi = v["a0s"]["ci95"]
        res["top_up"] = bool(lo <= MRB05 <= hi and lo <= 0 <= hi)
        res["a0s_point_prediction_frame"] = -1.1
    res["cap_hit"] = {a: float(np.mean([gens[a][f"{MODEL}/{a}/{s}/k{k}"]["status"] != "finished"
                                        for s in common for k in (0, 1)]))
                      for a in ["none"] + arms}
    for a in arms:
        if res["cap_hit"][a] > res["cap_hit"]["none"] + 0.05:
            keep = [i for i, s in enumerate(common) if all(
                gens[x][f"{MODEL}/{x}/{s}/k{k}"]["status"] == "finished"
                for x in ("none", a) for k in (0, 1))]
            res.setdefault("uncapped_beside", {})[a] = {"scenarios": len(keep), "est": round(adj(
                (base[0][keep], base[1][keep]), (arr[a][0][keep], arr[a][1][keep])), 4)}
    fp = set(json.loads(FINGERPRINT.read_text())["ids"])
    sub_w = {s: {c: w for c, w in ws.items() if f"{s}/{c}" in fp} for s, ws in weights.items()}
    fus = {}
    for rows in judge.values():
        fus |= A.unit_scores([r for r in rows if f"{r['unit'].split('/')[2]}/{r['criterion_id']}"
                              in fp], sub_w)
    fsids = [s for s in common if sub_w.get(s)]
    res["fingerprint_secondary"] = {}
    for a in ("a0s", "p_ctx", "p_ctx_n"):
        if a not in arms:
            continue
        d = np.array([np.mean([fus[f"{MODEL}/{a}/{s}/k{k}/response"]["credit_substring"]
                               - fus[f"{MODEL}/none/{s}/k{k}/response"]["credit_substring"]
                               for k in (0, 1)]) for s in fsids
                      if all(f"{MODEL}/{x}/{s}/k{k}/response" in fus for x in ("none", a)
                             for k in (0, 1))])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        res["fingerprint_secondary"][a] = {
            "scenarios": len(d), "raw_delta": round(float(d.mean()), 3),
            "ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}
    return res


def write(out: Path) -> dict:
    res = read(out)
    (out / "mrb2x2_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(">> mrb2x2:", json.dumps({k: res.get(k) for k in ("scenarios_read", "reading")}))
    return res


if __name__ == "__main__":
    write(Path(sys.argv[1]))
