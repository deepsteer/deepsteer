#!/usr/bin/env python3
"""Amendment 39 item 2: the lead's two quick reads on P-ctx, reproduced (zero GPU, saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/pctx_followups.py

(a) **The held-out legitimate orders split by an independent selection:** a held-out legitimate
scenario is *uncontested* when 2-roll's A0 followed the order on all 4 of its rollouts (the saved
action read, `outputs/pressure/act.jsonl`), else *contested*. On each part: P-ctx's second draw
against the verification pod's session base (held-out cell at k 4, its action read), the teacher
A0s's 2-roll follow share, and the same split by order type. Scenario bootstrap, 2,000 draws.

(b) **Cell A′ with its twins:** per item P(violating) of record (`s0a_reads.spec_rule`'s estimator)
on the 51 originals and their harm-reversed twins; each rung's shift against its own session base
on each, and the harm-conditioned part (the twins' shift minus the originals' shift), bootstrapped
over original-twin pairs. A0s's from S0a (against the saved A0); P-ctx from 2-ctx; the second draw
and P-ctx-f from the verification pod.

Writes `outputs/pctx_followups.json`.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pressure_pod as R  # noqa: E402
import cst_pressure_train_pod as T  # noqa: E402

CST = HERE.parent
P2 = CST / "outputs/pressure"
VER = CST / "outputs/pressure_verify"
B, SEED = 2000, "20261112|cst_pctx_followups"


def labels(path: Path) -> dict[str, str]:
    return {json.loads(x)["rollout_id"]: json.loads(x)["main"] for x in open(path)}


def follow(rows: list[dict], lab: dict, rid=lambda r: r["rollout_id"]) -> dict[str, list[int]]:
    u: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        v = lab.get(rid(r))
        if v in ("follows", "does_not"):
            u[r["scenario_id"]].append(int(v == "follows"))
    return dict(u)


def mean(u: dict, ids: list[str]) -> float | None:
    v = [statistics.mean(u[s]) for s in ids if u.get(s)]
    return statistics.mean(v) if v else None


def paired(a: dict, b: dict, ids: list[str]) -> dict:
    ids = [s for s in ids if a.get(s) and b.get(s)]
    rng = random.Random(SEED)
    bs = sorted(mean(a, d) - mean(b, d) for d in ([rng.choice(ids) for _ in ids]
                                                  for _ in range(B)))
    return {"delta": mean(a, ids) - mean(b, ids), "ci95": [bs[int(0.025 * B)],
                                                           bs[int(0.975 * B) - 1]], "n": len(ids)}


def part_a() -> dict:
    sc = {s["cell_id"]: s for s in R.scen_list(P2)}
    held = sorted(s for s in T.heldout_ids() if s in sc and sc[s]["kind"] == "legit")
    lab2 = labels(P2 / "act.jsonl")
    a0 = defaultdict(list)
    for r in R.rollouts(P2, "a0"):
        if r["scenario_id"] in held:
            a0[r["scenario_id"]].append(lab2.get(r["rollout_id"]))
    uncontested = sorted(s for s in held if len(a0[s]) == 4 and all(v == "follows" for v in a0[s]))
    contested = sorted(s for s in held if s not in uncontested)
    teacher = follow(R.rollouts(P2, "a0s"), lab2)
    labv = labels(VER / "heldout_act.jsonl")
    hv = {t: [json.loads(x) for x in open(VER / "heldout" / t / "chunk_000.jsonl")]
          for t in ("base", "p_ctx_draw2")}
    u = {t: follow(rows, labv) for t, rows in hv.items()}
    res = {"held_out_legit": len(held), "uncontested_rule": "A0 follows on 4 of 4 (2-roll)"}
    for name, ids in (("uncontested", uncontested), ("contested", contested), ("all", held)):
        res[name] = {"n": len(ids), "base": mean(u["base"], ids),
                     "p_ctx_draw2": mean(u["p_ctx_draw2"], ids),
                     "teacher_a0s_2roll": mean(teacher, ids),
                     "draw2_minus_base": paired(u["p_ctx_draw2"], u["base"], ids)}
    by: dict[str, list[str]] = defaultdict(list)
    for s in held:
        by[sc[s]["order_type"]].append(s)
    res["by_order_type"] = {k: {"n": len(v), "uncontested": sum(s in uncontested for s in v),
                                "base": mean(u["base"], v),
                                "p_ctx_draw2": mean(u["p_ctx_draw2"], v),
                                "teacher_a0s_2roll": mean(teacher, v)}
                            for k, v in sorted(by.items())}
    res["uncontested_ids"], res["contested_ids"] = uncontested, contested
    return res


def a2_twins(gen: Path, rung: str, base: str) -> dict:
    import cst_s0a_pod as S
    import numpy as np
    import s0a_reads as SR

    from deepsteer.kdg.schema import load_scenarios

    _, QS, A15, _ = SR._st()  # noqa: SLF001
    fst = {s.id: {o.option_id: o.norm_status for o in s.options}
           for f in S.FORCED for s in load_scenarios(f)[0]}

    def pr(tag: str) -> dict[str, float]:
        rr = []
        for f in sorted((gen / tag).glob("forced_plain_*.jsonl")):
            for x in open(f):
                r = json.loads(x)
                ns = fst[r["scenario_id"]][r["option"]] if r.get("option") else None
                rr.append({"scenario_id": r["scenario_id"],
                           "norm_status": None if r["hit_cap"] else ns,
                           "status": "truncated" if r["hit_cap"] else r["parse_status"]})
        return A15.p_record(rr)

    a, b = pr(base), pr(rung)
    origs = sorted(s for s in a if not s.endswith(QS.TWIN) and s in b and s + QS.TWIN in a
                   and s + QS.TWIN in b)
    so = np.array([b[s] - a[s] for s in origs])
    st = np.array([b[s + QS.TWIN] - a[s + QS.TWIN] for s in origs])
    rng = random.Random(SEED)
    idx = list(range(len(origs)))

    def ci(x: np.ndarray) -> list[float]:
        bs = sorted(float(np.mean(x[[rng.choice(idx) for _ in idx]])) for _ in range(B))
        return [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]

    return {"pairs": len(origs), "originals_shift": float(so.mean()), "originals_ci": ci(so),
            "twins_shift": float(st.mean()), "twins_ci": ci(st),
            "harm_conditioned": float((st - so).mean()), "harm_conditioned_ci": ci(st - so)}


def main() -> None:
    res = {"rule": "Amendment 39 item 2", "a_heldout_split": part_a(), "b_cell_a2_twins": {
        "a0s (S0a, vs the saved A0)": a2_twins(CST / "outputs/s0a/gen", "a0s", "a0"),
        "p_ctx (2-ctx)": a2_twins(P2 / "gen", "p_ctx", "base_session"),
        "p_ctx_draw2 (verification)": a2_twins(VER / "gen", "p_ctx_draw2", "base_session"),
        "p_ctx_f (verification)": a2_twins(VER / "gen", "p_ctx_f", "base_session")}}
    (CST / "outputs/pctx_followups.json").write_text(json.dumps(res, indent=1) + "\n")
    a = res["a_heldout_split"]
    for k in ("uncontested", "contested", "all"):
        d = a[k]["draw2_minus_base"]
        print(f"{k:11s} n {a[k]['n']:3d} base {a[k]['base']:.3f} draw2 {a[k]['p_ctx_draw2']:.3f} "
              f"teacher {a[k]['teacher_a0s_2roll']:.3f} | draw2-base {d['delta']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    for k, v in a["by_order_type"].items():
        print(f"  {k:26s} n {v['n']:2d} uncontested {v['uncontested']:2d} base {v['base']:.3f} "
              f"draw2 {v['p_ctx_draw2']:.3f} teacher {v['teacher_a0s_2roll']:.3f}")
    for k, v in res["b_cell_a2_twins"].items():
        print(f"{k:28s} orig {v['originals_shift']:+.3f} twins {v['twins_shift']:+.3f} "
              f"harm-conditioned {v['harm_conditioned']:+.3f} [{v['harm_conditioned_ci'][0]:+.3f}, "
              f"{v['harm_conditioned_ci'][1]:+.3f}] (pairs {v['pairs']})")


if __name__ == "__main__":
    main()
