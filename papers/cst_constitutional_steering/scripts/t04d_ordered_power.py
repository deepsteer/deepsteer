#!/usr/bin/env python3
"""CST Amendment 14 item 3: T0.4 re-run for the ordered-option share (zero GPU, saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t04d_ordered_power.py

The headline from G1 forward is the ordered-option share on the operator cell's 458 clear items
(Amendment 14 item 1). This re-runs T0.4's closed form on S0a's A0 and A0s rollouts (k 4 each) and
adds the two quantities Amendment 14 item 4 reads an arm on: the gain against A0s's band, and the
specificity cost on cell A′'s rule-right originals (bar 0.03).

Closed form (paired, scenario level, 80% power, two-sided alpha 0.05, MDE = 2.80 x SE), every
rollout with an action entering (no f thinning: the share is read on actions, not on tenets):
    w_arm   = E_s[p_s (1 - p_s)] k / (k - 1)        (unbiased within-scenario variance)
    het     = Var_s(d_s) - (w_A0 + w_A0s) / k       (effect heterogeneity of A0s - A0, measured)
    SE(arm - ref at k) = sqrt((w_arm + w_ref) / k + het_term) / sqrt(n)
Second derivations: (1) A0's k 4 split into two disjoint k 2 blocks, the scenario bootstrap SE
of block 1 - block 2 against the closed form at k 2 with het 0; (2) the scenario bootstrap SE of
A0s - A0 at k 4 against the closed form with the measured het.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402

CST = HERE.parent
SEED, N_BOOT, Z = 20261108, 10000, 2.80
KS = (2, 4, 6, 8)
SPEC_BAR = 0.03  # Amendment 14 item 4


def boot_se(d: np.ndarray, seed: int = SEED) -> float:
    idx = np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))
    return float(d[idx].mean(axis=1).std(ddof=1))


def w_of(per: dict[str, list[int]]) -> float:
    return float(np.mean([np.mean(v) * (1 - np.mean(v)) * len(v) / (len(v) - 1)
                          for v in per.values() if len(v) > 1]))


def ordered(lab: dict) -> dict[str, dict[str, list[int]]]:
    """Arm -> scenario -> per-rollout ordered-option indicator (actions only, capped excluded),
    the same units as ``s0a_reads.actions`` on panel|operator|clear_refuse."""
    out = {}
    for a in ("a0", "a0s"):
        u: dict[str, list[int]] = defaultdict(list)
        for r in A.load_jsonl(f"gen/{a}/panel_operator_*.jsonl"):
            if r["condition"] != "operator":  # the glob also matches operator_pr's chunks
                continue
            rec = lab["record"][r["scenario_id"]]
            if rec["stratum_of_record"] != "clear_refuse" or r["hit_cap"] or r["option"] is None:
                continue
            u[r["scenario_id"]].append(int(r["option"] == rec["ordered_option"]))
        out[a] = dict(u)
    return out


def headline(lab: dict) -> dict:
    u = ordered(lab)
    ids = sorted(set(u["a0"]) & set(u["a0s"]))
    n = len(ids)
    p = {a: np.array([np.mean(u[a][s]) for s in ids]) for a in u}
    w = {a: w_of({s: u[a][s] for s in ids}) for a in u}
    k_obs = {a: float(np.mean([len(u[a][s]) for s in ids])) for a in u}
    d = p["a0s"] - p["a0"]
    het = max(0.0, float(d.var(ddof=1)) - (w["a0"] / k_obs["a0"] + w["a0s"] / k_obs["a0s"]))

    def se(w1: float, w2: float, k: int, h: float) -> float:
        return sqrt((w1 + w2) / k + h) / sqrt(n)

    # second derivation 1: A0 null twin from disjoint k 2 blocks (rollout index 0,1 vs 2,3)
    twin = []
    for a in ("a0",):
        rows = A.load_jsonl(f"gen/{a}/panel_operator_*.jsonl")
        blk: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
        for r in rows:
            if r["condition"] != "operator":  # the glob also matches operator_pr's chunks
                continue
            rec = lab["record"][r["scenario_id"]]
            if rec["stratum_of_record"] != "clear_refuse" or r["hit_cap"] or r["option"] is None:
                continue
            blk[r["scenario_id"]][int(r["k"] >= 2)].append(
                int(r["option"] == rec["ordered_option"]))
        twin = np.array([np.mean(b[0]) - np.mean(b[1]) for b in blk.values() if b[0] and b[1]])
    tab = {}
    for k in KS:
        tab[f"k{k}"] = {
            "arm_vs_A0_null_mde": Z * se(w["a0"], w["a0"], k, 0.0),
            "arm_vs_A0s_null_mde": Z * se(w["a0s"], w["a0s"], k, 0.0),
            "arm_vs_A0s_mde_with_A0s_het": Z * se(w["a0s"], w["a0s"], k, het),
            "A0s_band_halfwidth_95": 1.96 * se(w["a0"], w["a0s"], k, het),
            "ratio_gain_mde": Z * se(w["a0s"], w["a0s"], k, het) / abs(float(d.mean())),
        }
    return {
        "n": n, "k_observed": k_obs, "share": {a: float(p[a].mean()) for a in p},
        "gain_A0s": float(d.mean()), "w": w, "het_A0s_minus_A0": het,
        "mde_table": tab,
        "second_derivation": {
            "A0_twin_k2_boot_se": boot_se(twin), "n_twin": int(len(twin)),
            "A0_twin_k2_closed_se": sqrt(2 * w["a0"] / 2) / sqrt(len(twin)),
            "A0s_minus_A0_k4_boot_se": boot_se(d),
            "A0s_minus_A0_k4_closed_se": se(w["a0"], w["a0s"], 4, het),
            "A0s_minus_A0_k4_closed_se_het0": se(w["a0"], w["a0s"], 4, 0.0),
        },
    }


def specificity() -> dict:
    """Cell A′'s rule-right originals: the shift's SE at the saved k 8 (from A0 and A0s's p_record
    per item) and what a 0.03 bar can resolve, under a point rule and a CI rule."""
    fr = json.loads((CST / "outputs" / "s0a_reads.json").read_text())["forced"]
    se_obs = (fr["a0s_vs_a0"]["original_shift"]["ci95"][1]
              - fr["a0s_vs_a0"]["original_shift"]["ci95"][0]) / (2 * 1.96)
    from math import erf

    def phi(x: float) -> float:
        return 0.5 * (1 + erf(x / sqrt(2)))

    out = {"n_pairs": fr["a0s_vs_a0"]["n_pairs"], "k": 8, "shift_se_observed": se_obs,
           "shift_mde": Z * se_obs, "bar": SPEC_BAR}
    for true in (0.0, 0.03, 0.088):
        out[f"true_{true}"] = {
            "P_pass_point_rule": phi((SPEC_BAR - true) / se_obs),
            "P_pass_ci_upper_below_bar": phi((SPEC_BAR - 1.96 * se_obs - true) / se_obs)}
    for k in (8, 16, 32):  # best case: SE falls as 1/sqrt(k); item heterogeneity floors it
        out[f"se_at_k{k}_best_case"] = se_obs * sqrt(8 / k)
    return out


def pilot(lab: dict) -> dict:
    """Amendment 20 item 3: the pilot read's MDE at k 2 on the operator cell (458 clear items), from
    A0's saved rollouts. A rung read at k 2 against A0 at its saved k 4, and against a control rung
    also at k 2; under the null each rung is A0-like (A0's w), and beside it the same with the
    heterogeneity A0s - A0 measured (a rung that moves varies by scenario). Second derivation:
    A0's own k 4
    split into two k 2 halves, the bootstrap SE of one half minus the other (rung vs control)."""
    h = headline(lab)
    n, w, het = h["n"], h["w"]["a0"], h["het_A0s_minus_A0"]

    def mde(var: float) -> float:
        return Z * sqrt(var / n)

    return {"n": n, "k_rung": 2, "w_a0": w,
            "rung_vs_A0_k4_null": mde(w / 2 + w / 4),
            "rung_vs_control_k2_null": mde(w / 2 + w / 2),
            "rung_vs_control_k2_with_het": mde(w / 2 + w / 2 + het),
            "second_derivation_half_split_se_x_2.8":
                Z * h["second_derivation"]["A0_twin_k2_boot_se"],
            "note": "A0 at k 4 is the saved baseline; rungs and controls generate at k 2"}


def main() -> None:
    lab = A.labels()
    h = headline(lab)
    # most probable failure: reading a different cell or unit than the report's headline (the
    # panel_operator_* glob also matches operator_pr); the shares must equal s0a_reads' actions
    rd = json.loads((CST / "outputs" / "s0a_reads.json").read_text())
    ref = rd["actions"]["panel|operator|clear_refuse"]["ordered"]
    assert all(abs(h["share"][a] - ref[a]) < 1e-9 for a in ("a0", "a0s")), (h["share"], ref)
    res = {"rule": "CST Amendment 14 item 3", "headline": h, "pilot_k2": pilot(lab),
           "specificity": specificity()}
    (CST / "outputs" / "t04d_ordered_power.json").write_text(json.dumps(res, indent=1) + "\n")
    h = res["headline"]
    print("n", h["n"], "share", {a: round(v, 3) for a, v in h["share"].items()}, "gain",
          round(h["gain_A0s"], 3), "w", {a: round(v, 4) for a, v in h["w"].items()}, "het",
          round(h["het_A0s_minus_A0"], 4))
    for k, v in h["mde_table"].items():
        print(k, {m: round(x, 4) for m, x in v.items()})
    print("2nd", {m: round(x, 4) for m, x in h["second_derivation"].items()})
    print("spec", json.dumps(res["specificity"], indent=None))
    pk = res["pilot_k2"]
    print("pilot", {k: round(v, 4) if isinstance(v, float) else v for k, v in pk.items()})


if __name__ == "__main__":
    main()
