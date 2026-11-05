#!/usr/bin/env python3
"""CST Amendment 1's zero-GPU inputs (author's G0 decisions D7, D9, D10 and addition 1).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t04b_amendment1.py

Writes ``outputs/t04b_amendment1.json``:
- D7: the share of root tenets (v1 roots of the v2 sets) that the document-internal T_conf plus a
  panel "any dissent" rule would exclude from training, projected from the measured agreement of
  the matching task (the only panel-agreement data before S0), with the tenet-rate rule beside it.
- D9: MoReBench matched-length MDE (MRB v0.9.3 Δ_adj) at k 4 on 500, from the P2 rows (k 2):
  the per-rollout variance from `none`'s k0-vs-k1 difference is rescaled from k 2 to k 4.
- D10: the disjoint capability pool and the paired difference-CI MDE at n 200 to 800.
- Addition 1: panel MDE on ΔS for user at k 2 and operator_pr at k 6, and the rollout budget.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]

import t04_power as P  # noqa: E402

from deepsteer.constitutions import TenetSet  # noqa: E402

CST = REPO / "papers" / "cst_constitutional_steering"
Z = P.Z


# ---- D7 -------------------------------------------------------------------------------------

def _roots(ids) -> set[str]:
    return {i.rstrip("bcdefgh") for i in ids}


def _doc_conf_roots(d: str) -> tuple[set[str], set[str]]:
    from deepsteer.constitutions import document_conflicts

    ts = TenetSet.load(CST / "data" / f"tenets_{d}_v2.json")
    r = json.loads((CST / "outputs" / "t01_conflicts_v2.json").read_text())[d]
    ids = set(ts.ids())
    by = ts.by_id()
    for c in r["conditioned_on"]:
        if c["id"] in ids and c["on"] in ids:
            by[c["id"]].conditioned_on.append(c["on"])
    comp = [tuple(c["pair"].split("|")) for c in r["competing"]]
    comp = [p for p in comp if all(x in ids for x in p)]
    return _roots(document_conflicts(ts, comp)), _roots(ids)


def _e_from_agreement(a: float) -> float:
    """Per-judge deviation rate e from pairwise agreement a = (1-e)^2 + e^2 (binary labels)."""
    return (1 - sqrt(2 * a - 1)) / 2


def d7() -> dict:
    v = json.loads((CST / "outputs" / "t01_tcore_votes.json").read_text())["votes"]
    both = [x for x in v.values() if "claude" in x and "codex" in x]
    a_cross = float(np.mean([x["claude"] == x["codex"] for x in both]))
    rt = json.loads((CST / "outputs" / "t01_retest.json").read_text())
    agreements = {"cross_family_matching_v1": a_cross, "claude_retest_matching": rt["agreement"],
                  "hypothetical_0.95": 0.95}
    n_scen = 586 + 132 + 102  # panel, H1_held, cell A′ forced: the scored S sets
    out: dict = {"model": "three judges, binary application label, each deviating from the "
                          "consensus independently with rate e; agreement a = (1-e)^2 + e^2; an "
                          "application is unanimous with probability u = (1-e)^3 + e^3; a root "
                          "with m applications is tagged under any-dissent with 1 - u^m; "
                          "m = n_scenarios x shortlist / roots",
                 "agreements": agreements, "n_scenarios_scored": n_scen, "docs": {}}
    for d in ("C_anth", "C_spec"):
        conf, roots = _doc_conf_roots(d)
        R = len(roots)
        doc_share = len(conf) / R
        grid = {}
        for name, a in agreements.items():
            e = _e_from_agreement(a)
            u = (1 - e) ** 3 + e ** 3
            row = {}
            for L in (3, 5, 8):
                m = n_scen * L / R
                p_tag = 1 - u ** m
                row[f"shortlist{L}"] = {"m_per_root": round(m, 1), "panel_any_dissent": p_tag,
                                        "union_with_doc": 1 - (1 - doc_share) * (1 - p_tag)}
            # tenet-rate rule: tag a root when its dissent rate exceeds the 0.9 quantile of a
            # noise-only binomial(m, 1-u) rate; by construction about 10% under noise alone
            grid[name] = {"e": e, "unanimity_u": u, **row}
        out["docs"][d] = {"roots": R, "doc_internal_roots": len(conf), "doc_share": doc_share,
                          "grid": grid}
    worst = max(g["shortlist5"]["union_with_doc"] for dd in out["docs"].values()
                for g in dd["grid"].values())
    out["flag_over_25pct"] = worst > 0.25
    out["rate_rule"] = ("a root enters T_conf when its dissent rate over its applications exceeds "
                        "the 0.9 quantile of the noise-only rate (1 - u, from the S0 calibration "
                        "items); under noise alone it tags about 10% of roots by construction, "
                        "so the union with the document set stays near doc_share + 0.1")
    return out


# ---- D9 -------------------------------------------------------------------------------------

def d9() -> dict:
    sys.path.insert(0, str(REPO / "papers/mrb_morebench/scripts"))
    import mrb_g1_analyze as A  # noqa: E402
    import mrb_p2_length as L  # noqa: E402

    from deepsteer.datasets.morebench import load_morebench  # noqa: E402

    out_dir = REPO / "papers/mrb_morebench/outputs/p2"
    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    us = A.unit_scores(A.read_jsonl(P.src(out_dir / "judge/shard0.jsonl")), weights)
    gens = {}
    for f in sorted((out_dir / "gen").glob("*.jsonl")):
        gens.update({r["job_id"]: r for r in A.read_jsonl(P.src(f))})
    sids = [it.scenario_id for it in items]
    rng = np.random.default_rng(P.SEED)
    res: dict = {"estimator": "MRB v0.9.3 Δ_adj; k 2 measured, k 4 by rescaling the per-rollout "
                              "variance (σ²_w from none's k0 − k1 raw difference: var = 2σ²_w; "
                              "under the arm the same σ²_w is assumed)",
                 "bar_of_record": -0.87}
    for m in L.MODELS:
        d = L.arrays(us, gens, m, sids)
        full = [L.estimates(d, rng.integers(0, 500, 500))["delta_adj"] for _ in range(1000)]
        se2 = float(np.std(full, ddof=1))
        s0 = d["none"][0]
        sw2 = float(np.var(s0[:, 1] - s0[:, 0], ddof=1) / 2)  # per-rollout variance
        # Var(Δ_s) at k: between + 2σ²_w/k; at k 2 the rollout part is σ²_w
        var_between = max(0.0, se2 ** 2 * 500 - sw2)
        se = {k: sqrt((var_between + 2 * sw2 / k) / n) for k in (2, 4) for n in (500,)}
        res[m] = {"se_k2_500": se2, "sigma2_w": sw2, "var_between_scenario": var_between,
                  "mde_k2_500": Z * se2,
                  "mde_k4_500": Z * sqrt((var_between + 2 * sw2 / 4) / 500),
                  "mde_k4_200": Z * sqrt((var_between + 2 * sw2 / 4) / 200),
                  "rollout_share_of_variance_k2": sw2 / (var_between + sw2) if var_between + sw2
                  else None, "se_check": se}
    return res


# ---- D10 ------------------------------------------------------------------------------------

def d10() -> dict:
    cap = P.capability()
    disc = float(np.mean([cap[a]["discordant_share"] for a in ("care", "base", "care2")]))
    gate = json.loads((REPO / "papers/st_live_reference/data/t15_capability_v1.json").read_text())
    sel = json.loads((REPO / "papers/st_live_reference/data/st2qp_selection_v1.json").read_text())
    test_sizes = {"openai/gsm8k": 1319, "cais/mmlu": 14042}  # test splits (HF dataset cards)
    used = {k: sum(i["dataset"] == k for i in gate["items"] + sel["items"]) for k in test_sizes}
    free = {k: test_sizes[k] - used[k] for k in test_sizes}
    rows = {n: {"se_paired": sqrt(disc / n), "mde": Z * sqrt(disc / n)}
            for n in (200, 400, 600, 800)}
    return {"discordant_share_mean": disc, "accuracy_none": cap["accuracy_none"],
            "pool_test_sizes": test_sizes, "used_by_st": used, "disjoint_free": free,
            "by_n": rows, "note": "SE from the paired discordance at a near-zero mean difference "
                                  "(sqrt(disc/n)); the gate's 200 are ST's and stay disjoint"}


# ---- addition 1 -----------------------------------------------------------------------------

def addition1() -> dict:
    w_h1, w_worst = 0.098, 0.25  # T04_POWER §1: H1_held thinking-on w; worst case
    n = 586
    cells = {"user_k2": 2, "user_k4": 4, "operator_pr_k4": 4, "operator_pr_k6": 6}
    mde = {c: {f"f{f}": {"h1_w": P.mde_closed(n, k, f, w_h1), "worst_w": P.mde_closed(n, k, f,
                                                                                    w_worst)}
               for f in (0.3, 0.5, 0.8)} for c, k in cells.items()}
    plan_d2 = {"plain": 4, "user": 4, "operator": 4, "operator_pr": 4}
    plan_a1 = {"plain": 4, "user": 2, "operator": 4, "operator_pr": 6}
    rate = 480 / 0.15  # ST Amendment 59: 480 thinking-on rollouts (medium, cap 1,536) per 0.15
    tot = {name: sum(p.values()) * n for name, p in (("d2", plan_d2), ("addition1", plan_a1))}
    return {"mde_dS_panel": mde, "rollouts_per_arm": tot,
            "rollouts_per_a100h_basis": rate,
            "a100h_per_arm": {k: v / rate for k, v in tot.items()},
            "note": "moving user from k 4 to k 2 and operator_pr from k 4 to k 6 keeps the panel "
                    "at 16 rollouts per scenario per arm: zero extra cost against D2 as accepted"}


def main() -> None:
    res = {"d7": d7(), "d9": d9(), "d10": d10(), "addition1": addition1()}
    (CST / "outputs" / "t04b_amendment1.json").write_text(json.dumps(res, indent=1) + "\n")
    for d, x in res["d7"]["docs"].items():
        print("D7", d, "roots", x["roots"], "doc", x["doc_internal_roots"],
              f"{x['doc_share']:.3f}")
        for name, g in x["grid"].items():
            print("   ", name, f"e={g['e']:.3f}", {k: round(v["union_with_doc"], 3)
                                                  for k, v in g.items() if k.startswith("short")})
    print("D7 flag >25%:", res["d7"]["flag_over_25pct"])
    for m, x in res["d9"].items():
        if isinstance(x, dict):
            print("D9", m, {k: round(v, 3) for k, v in x.items() if k.startswith("mde")})
    print("D10 pool", res["d10"]["disjoint_free"], {n: round(r["mde"], 4)
                                                     for n, r in res["d10"]["by_n"].items()})
    a = res["addition1"]
    print("A1", {c: round(v["f0.5"]["h1_w"], 4) for c, v in a["mde_dS_panel"].items()},
          a["rollouts_per_arm"], {k: round(v, 2) for k, v in a["a100h_per_arm"].items()})


if __name__ == "__main__":
    main()
