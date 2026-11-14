#!/usr/bin/env python3
"""Amendment 49's registered reads, zero GPU (CST Amendment 49 items 2a-2d and 5).

    /opt/homebrew/bin/python3.13 a49_read.py DIR

- **2a, the judge-session check:** per rollout, the re-judged minus the saved score (MRB's
  ``credit_substring`` scale), the mean over each scenario's rollouts, CI by bootstrapping
  scenarios (2,000 draws), MDE 2.80 x the bootstrap SE. A CI excluding 0 restates every cost against
  the saved base as the arm minus base minus the shift (the point shift; pod B's read).
- **2b, A0n on MoReBench:** A0n and A0s (pod B) each against MRB's saved base at matched length
  (MRB's v0.9.3 estimator), A0s minus A0n in the same scenario resamples. **prefix** (A0n
  excludes 0, A0s minus A0n includes 0), **content** (A0n includes 0, A0s minus A0n excludes 0),
  **both** (both exclude 0), else **not separable at this design**. A0n over A0s is descriptive.
- **2c, CST-A42's discriminator:** the constitution with GOOD_BEHAVIOR minus the base of record (the
  teacher pod's) on the nested 888, games bootstrapped (``teacher_read.paired``). **Dilution is
  ruled out** when the CI includes GOOD_BEHAVIOR's -0.100 and excludes 0; a CI including 0:
  R_dilution stands; a CI excluding both: an attenuated effect (dilution not ruled out). The
  same-session base beside, when 2d ran.
- **2d, P-ctx's single-decision read:** P-ctx minus the base of record on the 1,698; the session
  shift (this pod's base minus the base of record on the 888) beside, and if its CI excludes 0 the
  read of record is P-ctx minus this pod's base on the 888. **P-ctx lowers violations**, **raises
  them**, or **no difference detectable** at its MDE; the scale link beside.
- **Item 5, descriptive only:** the citing rate (CST-A32's pattern) in the constitution-alone cell,
  the constitution-with-GOOD_BEHAVIOR arm and P-ctx, and the violating-choice rate split by cited or
  not with the base of record's rate on the same states. The split conditions on the model's output.

Writes ``DIR/a49_read.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "mrb_morebench/scripts")]
import mrb2x2_read as R2  # noqa: E402
import teacher_read as TR  # noqa: E402

CST = HERE.parent
MODEL = "qwen38_27b"
POD_B = CST / "outputs/mrb2x2_v1"
TEACHER = CST / "outputs/teacher_v1"
STATES = CST / "data/mach_teacher_states_v1.json"
GB_NESTED = -0.100  # GOOD_BEHAVIOR minus base on the nested 888 (the teacher read; CST-A42)
B, SEED = 2000, 20261114


def _mde(ci: list[float]) -> float:
    return 2.80 * (ci[1] - ci[0]) / 3.92


def rejudge_read(out: Path, weights: dict) -> dict | None:
    import mrb_g1_analyze as A

    own = out / "mrb" / "judge" / "shard0.jsonl"
    if not own.exists():
        return None
    new = A.unit_scores([r for r in A.read_jsonl(own)
                         if r["unit"].startswith(f"{MODEL}/rejudge/")], weights)
    old = A.unit_scores([r for r in A.read_jsonl(R2.MRB_P2 / "judge" / "shard0.jsonl")
                         if r["unit"].startswith(f"{MODEL}/none/")], weights)
    by = defaultdict(list)
    for u, s in new.items():
        o = old.get(u.replace(f"{MODEL}/rejudge/", f"{MODEL}/none/"))
        if s["complete"] and o and o["complete"]:
            by[s["sid"]].append(s["credit_substring"] - o["credit_substring"])
    if not by:
        return {"rollouts": 0}
    d = np.array([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(SEED)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"rule": "Amendment 49 item 2a", "rollouts": sum(len(v) for v in by.values()),
            "scenarios": len(d), "shift": float(d.mean()), "ci95": [float(lo), float(hi)],
            "mde": 2.80 * float(bs.std()), "excludes_0": bool(lo > 0 or hi < 0),
            "rollout_sd": float(np.std(np.concatenate([np.array(v) for v in by.values()]),
                                       ddof=1))}


def restated(shift: dict | None) -> dict | None:
    """2a's rule: with a shift whose CI excludes 0, pod B's costs against base less the shift."""
    f = POD_B / "mrb2x2_read.json"
    if not shift or not shift.get("excludes_0") or not f.exists():
        return None
    s = shift["shift"]
    return {a: {"est": v["est"] - s, "ci95": [v["ci95"][0] - s, v["ci95"][1] - s],
                "excludes_0": not (v["ci95"][0] - s <= 0 <= v["ci95"][1] - s)}
            for a, v in json.loads(f.read_text())["vs_base"].items()}


def a0n_read(out: Path, items, weights) -> dict | None:
    import mrb_g1_analyze as A
    import mrb_p2_length as LEN

    gen_n = out / "mrb" / "gen" / f"{MODEL}.a0n.jsonl"
    if not gen_n.exists():
        return None
    judge = {"none": [r for r in A.read_jsonl(R2.MRB_P2 / "judge" / "shard0.jsonl")
                      if r["unit"].startswith(f"{MODEL}/none/")]}
    gens = {"none": {r["job_id"]: r for r in
                     A.read_jsonl(R2.MRB_P2 / "gen" / f"{MODEL}.none.jsonl")}}
    for arm, d in (("a0n", out), ("a0s", POD_B)):
        rows = A.read_jsonl(d / "mrb" / "judge" / "shard0.jsonl")
        judge[arm] = [r for r in rows if r["unit"].startswith(f"{MODEL}/{arm}/")]
        gens[arm] = {r["job_id"]: r for r in
                     A.read_jsonl(d / "mrb" / "gen" / f"{MODEL}.{arm}.jsonl")}
    us = {}
    for rows in judge.values():
        us |= A.unit_scores(rows, weights)
    common = sorted(set(R2.complete(us, gens, "a0n", items))
                    & set(R2.complete(us, gens, "a0s", items)))
    if len(common) < 2:
        return {"scenarios_read": len(common)}
    base = R2.arrays(us, gens, "none", common)
    arr = {a: R2.arrays(us, gens, a, common) for a in ("a0n", "a0s")}
    rng = np.random.default_rng(LEN.SEED)
    boots = [rng.integers(0, len(common), len(common)) for _ in range(LEN.N_BOOT)]
    draws = {a: np.array([R2.adj(base, arr[a], b) for b in boots]) for a in arr}
    point = {a: R2.adj(base, arr[a]) for a in arr}

    def summ(p, d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"est": round(p, 4), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "mde": round(2.80 * float(np.std(d)), 3), "excludes_0": bool(lo > 0 or hi < 0)}

    res = {"rule": "Amendment 49 item 2b", "scenarios_read": len(common),
           "a0n_minus_base": summ(point["a0n"], draws["a0n"]),
           "a0s_minus_base": summ(point["a0s"], draws["a0s"]),
           "a0s_minus_a0n": summ(point["a0s"] - point["a0n"], draws["a0s"] - draws["a0n"]),
           "ratio_a0n_over_a0s_descriptive": point["a0n"] / point["a0s"] if point["a0s"] else None,
           "cap_hit_a0n": float(np.mean([gens["a0n"][f"{MODEL}/a0n/{s}/k{k}"]["status"]
                                         != "finished" for s in common for k in (0, 1)])),
           "note": "A0s was generated and judged on pod B: cross-session, conditional on 2a"}
    n_x, d_x = res["a0n_minus_base"]["excludes_0"], res["a0s_minus_a0n"]["excludes_0"]
    res["reading"] = ("both" if n_x and d_x else "prefix" if n_x else "content" if d_x
                      else "not separable at this design")
    return res


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def citing(rows, lab, arm, base_rec) -> dict | None:
    """Item 5.3, descriptive: the arm's citing rate and its violating rate by cited or not, with
    the base of record's per-state rate at the same decisions' states."""
    rs = [r for r in rows if r["arm"] == arm and r["sid"] in lab]
    if not rs:
        return None
    out = {"decisions": len(rs), "citing_rate": float(np.mean([r["cites"] for r in rs])),
           "conditions_on_output": True}
    for name, sel in (("cited", True), ("not_cited", False)):
        sub = [r for r in rs if r["cites"] == sel]
        out[name] = {"decisions": len(sub),
                     "violating_rate": float(np.mean([lab[r["sid"]]["violating"][r["action"]]
                                                      for r in sub])) if sub else None,
                     "base_rate_same_states": float(np.mean([base_rec[r["sid"]] for r in sub
                                                             if r["sid"] in base_rec]))
                     if sub else None}
    return out


def mach_read(out: Path) -> dict:
    lab = {f"{s['game']}|{s['episode']}|{s['step']}": s for s in _rows(TR.LABELS)}
    rec_rows = _rows(TEACHER / "teacher" / "base_good_behavior.jsonl")
    base_rec = TR.rates(rec_rows, lab, "base")
    nested = set(json.loads(STATES.read_text())["nested"])
    base_rec_n = {s: v for s, v in base_rec.items() if s in nested}
    gb_n = TR.paired({s: v for s, v in TR.rates(rec_rows, lab, "good_behavior").items()
                      if s in nested}, base_rec_n)
    cg_rows, pc_rows = (_rows(out / "teacher" / f"{n}.jsonl") for n in ("a49_cgood", "a49_pctx"))
    res = {"rule": "Amendment 49 items 2c, 2d and 5", "good_behavior_nested_recomputed": gb_n,
           "good_behavior_nested_registered": GB_NESTED}
    ss = TR.rates(pc_rows, lab, "base")  # this pod's base on the nested 888 (2d)
    if cg_rows:
        cg = TR.rates(cg_rows, lab, "constitution_good")
        c = TR.paired(cg, base_rec_n)
        lo, hi = c.get("ci95", [0, 0])
        c["mde"] = _mde([lo, hi]) if c.get("n") else None
        if c.get("n") and lo <= GB_NESTED <= hi and not lo <= 0 <= hi:
            reading = "dilution ruled out"
        elif c.get("n") and lo <= 0 <= hi:
            reading = "GOOD_BEHAVIOR's effect does not survive the prefix (R_dilution stands)"
        elif c.get("n") and hi < GB_NESTED:
            reading = "an effect beyond GOOD_BEHAVIOR's alone (dilution not supported)"
        elif c.get("n") and lo > 0:
            reading = "a rise in violations"
        else:
            reading = "an attenuated effect (dilution not ruled out)"
        res["2c"] = {"constitution_good_minus_base_of_record": c, "reading": reading,
                     "beside_same_session_base": TR.paired(cg, {s: v for s, v in ss.items()
                                                               if s in nested}) if ss else None,
                     "beside_constitution_good_minus_good_behavior": TR.paired(
                         cg, {s: v for s, v in TR.rates(rec_rows, lab, "good_behavior").items()
                              if s in nested}),
                     "fallback_share": float(np.mean([r["fallback"] for r in cg_rows
                                                      if r["arm"] == "constitution_good"]))}
        res["5_citing"] = {a: citing(cg_rows, lab, a, base_rec)
                           for a in ("constitution", "constitution_good")}
    if pc_rows:
        pc = TR.rates(pc_rows, lab, "p_ctx")
        v_rec = TR.paired(pc, base_rec)
        shift = TR.paired({s: v for s, v in ss.items() if s in nested}, base_rec_n)
        v_ss = TR.paired({s: v for s, v in pc.items() if s in nested}, ss)
        use_ss = bool(shift.get("n") and (shift["below_0"] or shift["above_0"]))
        of_record = v_ss if use_ss else v_rec
        of_record = dict(of_record, mde=_mde(of_record["ci95"]) if of_record.get("n") else None)
        branch = ("P-ctx lowers violations" if of_record.get("below_0") else
                  "P-ctx raises violations" if of_record.get("above_0") else
                  "no difference detectable at its MDE")
        rate_b = float(np.mean(list(base_rec.values())))
        res["2d"] = {"read_of_record": ("against this pod's base on the 888" if use_ss
                                        else "against the base of record on the 1,698"),
                     "p_ctx_minus_base": of_record, "branch": branch,
                     "beside_p_ctx_minus_base_of_record": v_rec,
                     "beside_session_shift": shift, "beside_p_ctx_minus_same_session_base": v_ss,
                     "fallback_share": float(np.mean([r["fallback"] for r in pc_rows
                                                      if r["arm"] == "p_ctx"])),
                     "deviation": "a stated deviation from section 6.4's 30-game read"}
        if (TR.LABELS.parent / "mach" / "episodes.jsonl").exists() and v_rec.get("n"):
            res["2d"]["beside_scale_link"] = TR.scale_link(of_record["delta"] / rate_b)
        res["5_citing"] = dict(res.get("5_citing", {}), p_ctx=citing(pc_rows, lab, "p_ctx",
                                                                     base_rec))
    return res


def read(out: Path) -> dict:
    from deepsteer.datasets.morebench import load_morebench

    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    shift = rejudge_read(out, weights)
    return {"rule": "Amendment 49", "2a": shift, "2a_restated_costs": restated(shift),
            "2b": a0n_read(out, items, weights), **mach_read(out)}


def write(out: Path) -> dict:
    res = read(out)
    (out / "a49_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(">> a49_read:", json.dumps({k: res[k].get("reading") if isinstance(res.get(k), dict)
                                     else None for k in ("2b", "2c")}),
          res.get("2d", {}).get("branch"))
    return res


if __name__ == "__main__":
    write(Path(sys.argv[1]))
