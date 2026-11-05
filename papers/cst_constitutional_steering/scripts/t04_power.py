#!/usr/bin/env python3
"""CST_PLAN T0.4: power table from saved arrays (zero GPU, zero API).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t04_power.py

Reads only saved per-rollout / per-scenario arrays from the ST, MRB and MCH lines; every source is
recorded with its sha256. Writes ``papers/cst_constitutional_steering/outputs/t04_power.json``.

S (CST_PLAN §1) does not exist yet. Its proxy here is the per-scenario departure share under
thinking-on generation on Qwen3.8-27B under ``none``: the share of parsed rollouts whose action is
the norm-``violating`` option (on H1 that is the protective departure from the written rule; on the
panel's rule-right items and cell A′'s forced items it is breaking the rule). S itself is
conditional on a rollout having an applied tenet, so a fraction ``f`` of rollouts enter it; the
closed form below carries that thinning.

Closed form (paired, scenario level, 80% power, two-sided alpha 0.05, MDE = 2.80 x SE):
    d_s = S_arm(s) - S_none(s);   Var(d_s | null) = 2 w E[1/m | m > 0] + 2 tau2
    w     = E_s[p_s (1 - p_s)]        (within-scenario Bernoulli variance, unbiased from k rollouts)
    m     ~ Binomial(k, f)            (rollouts with an applied tenet)
    tau2  = per-scenario run-to-run excess variance (from the none-vs-none twin; 0 if none)
    n_eff = n (1 - (1 - f)^k)^2       (scenarios with >= 1 applied rollout in both arms)
    MDE   = 2.80 sqrt(Var(d_s) / n_eff)
Second derivation: the empirical scenario bootstrap of the observed none-vs-none twin differences
on H1_held at k = 8, 4 and 2 (rollouts split into disjoint blocks), against the closed form.
"""

from __future__ import annotations

import hashlib
import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ST = REPO / "papers/st_live_reference"
OUT = REPO / "papers/cst_constitutional_steering/outputs"
Q = ST / "outputs"
SEED, N_BOOT = 20261104, 10000
Z = 2.80  # 1.96 + 0.84
KS, FS = (2, 4, 8), (0.3, 0.5, 0.8, 1.0)
SOURCES: dict[str, str] = {}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def src(p: Path) -> Path:
    SOURCES[str(p.relative_to(REPO))] = _sha(p)
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in src(p).read_text().splitlines() if x.strip()]


def jload(p: Path):
    return json.loads(src(p).read_text())


def boot(x, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    d = x[np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "n": int(len(x)),
            "se_boot": float(d.std(ddof=1)), "mde_boot": float(Z * d.std(ddof=1)),
            "point_in_ci": bool(lo <= x.mean() <= hi)}


# --------------------------------------------------------------------------------------------
# per-scenario departure shares and variance components
# --------------------------------------------------------------------------------------------


def by_scenario(rows: list[dict], ids: set[str] | None = None) -> dict[str, list[int]]:
    """Per scenario, the parsed rollouts' departure indicators in rollout order (unparsed and
    truncated rollouts are missing, as S is over parsed actions)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        s = r["scenario_id"]
        if ids is not None and s not in ids:
            continue
        ns = r.get("norm_status")
        if ns is None or ns == "None":
            continue
        out.setdefault(s, []).append((int(r["rollout"]), int(ns == "violating")))
    return {s: [v for _, v in sorted(x)] for s, x in out.items()}


def components(b: dict[str, list[int]]) -> dict:
    """Mean departure, within-scenario Bernoulli variance w (unbiased), between-scenario variance
    of the true rate, and the truncation-free k actually observed."""
    ps, ws, ks = [], [], []
    for v in b.values():
        k = len(v)
        p = float(np.mean(v))
        ps.append(p)
        ks.append(k)
        ws.append(p * (1 - p) * k / (k - 1) if k > 1 else 0.0)
    ps, ws, ks = np.array(ps), np.array(ws), np.array(ks)
    w = float(ws.mean())
    between = float(max(0.0, ps.var(ddof=1) - np.mean(ws / ks)))
    return {"n_scenarios": int(len(ps)), "mean_departure": float(ps.mean()), "w": w,
            "between_var_true_rate": between, "mean_k_parsed": float(ks.mean()),
            "share_scenarios_at_0": float(np.mean(ps == 0)),
            "share_scenarios_at_1": float(np.mean(ps == 1))}


def e_inv_m(k: int, f: float) -> tuple[float, float]:
    """E[1/m | m > 0] for m ~ Binomial(k, f), and P(m > 0)."""
    pk = [comb(k, m) * f ** m * (1 - f) ** (k - m) for m in range(k + 1)]
    p0 = 1 - pk[0]
    return sum(pk[m] / m for m in range(1, k + 1)) / p0, p0


def mde_closed(n: int, k: int, f: float, w: float, tau2: float = 0.0) -> float:
    einv, p_any = e_inv_m(k, f)
    var_d = 2 * w * einv + 2 * tau2
    return float(Z * sqrt(var_d / (n * p_any ** 2)))


def mde_grid(ns: dict[str, int], w: float, tau2: float) -> dict:
    return {nm: {f"k{k}": {f"f{f}": round(mde_closed(n, k, f, w, tau2), 4) for f in FS}
                 for k in KS} for nm, n in ns.items()}


# --------------------------------------------------------------------------------------------
# 1. S proxy: H1_held twin (none vs none, two independent generation runs)
# --------------------------------------------------------------------------------------------


def s_proxy() -> dict:
    split = jload(ST / "split_v1.json")
    held = set(split["H1_held"])
    rec = jload(Q / "st2qp/qwen38_27b/st2qp_record.json")
    engaged = set(rec["phases"]["base_hf"]["engaged_held"])
    re_rows = jsonl(Q / "st2qp/qwen38_27b/gen/h1_none.jsonl")  # vllm_bf16, stage st2qp
    sel_rows = jsonl(Q / "st2q/qwen38_27b/gen/h1_rollouts.jsonl")  # vllm, stage st2q
    A, B = by_scenario(re_rows, held), by_scenario(sel_rows, held)
    trunc = float(np.mean([r.get("status") == "truncated" for r in re_rows]))
    parse = float(np.mean([r.get("norm_status") not in (None, "None") for r in re_rows]))
    comp = components(A)
    ids = sorted(set(A) & set(B))
    d8 = [np.mean(A[s]) - np.mean(B[s]) for s in ids]
    w_pool = float(np.mean([components({s: A[s]})["w"] + components({s: B[s]})["w"]
                            for s in ids]) / 2)
    kA = np.array([len(A[s]) for s in ids])
    kB = np.array([len(B[s]) for s in ids])
    pred_var = float(np.mean([w_pool / a + w_pool / b for a, b in zip(kA, kB)]))
    obs_var = float(np.var(d8, ddof=1))
    tau2 = max(0.0, (obs_var - pred_var) / 2)

    # empirical second derivation: disjoint rollout blocks of size k from each run
    emp = {}
    for k in KS:
        nb = 8 // k
        sds, means = [], []
        for j in range(nb):
            # k = 8 uses every parsed rollout (d8); smaller blocks need a full block in both runs
            dd = d8 if k == 8 else [
                np.mean(A[s][j * k:(j + 1) * k]) - np.mean(B[s][j * k:(j + 1) * k])
                for s in ids if len(A[s]) >= (j + 1) * k and len(B[s]) >= (j + 1) * k]
            bb = boot(dd, SEED + j)
            sds.append(bb["se_boot"])
            means.append(bb["mean"])
        emp[f"k{k}"] = {"blocks": nb, "mde_empirical": round(Z * float(np.mean(sds)), 4),
                        "mean_twin_diff_by_block": [round(x, 4) for x in means],
                        "mde_closed_form_f1": round(mde_closed(len(ids), k, 1.0, w_pool, tau2), 4),
                        "mde_closed_form_f1_no_run_term": round(
                            mde_closed(len(ids), k, 1.0, w_pool, 0.0), 4)}
    # split-half within the st2qp run (rollouts 0-3 vs 4-7): a matched null with no run term
    half = [np.mean(A[s][:4]) - np.mean(A[s][4:8]) for s in A if len(A[s]) >= 8]

    # sets for the variance components (thinking-on, none)
    spec_ids = set(jload(ST / "data/t15_specificity_v1.json")["ids"])
    sets = {
        "H1_held_132": comp,
        "H1_held_engaged_17": components({s: A[s] for s in A if s in engaged}),
        "panel_rule_right_60": components(by_scenario(
            jsonl(Q / "st2qg/qwen38_27b/gen/spec_none.jsonl"), spec_ids)),
        "cellA2_forced_102": components(by_scenario(
            jsonl(Q / "st2qp/qwen38_27b/gen/forced_none.jsonl"))),
    }
    return {
        "proxy": "departure share (norm_status == violating among parsed rollouts), thinking on at "
                 "effort medium, cap 1,536, T 0.7, Qwen3.8-27B, arm none",
        "twin": {"runs": ["st2q h1_rollouts (engine vllm; the selection sample)",
                          "st2qp h1_none (engine vllm_bf16; the independent re-run)"],
                 "n_scenarios": len(ids), "diff_k8": boot(d8),
                 "sd_of_per_scenario_diff": float(np.std(d8, ddof=1)),
                 "predicted_sd_binomial_only": float(sqrt(pred_var)),
                 "tau2_run_excess_per_arm": tau2,
                 "w_pooled": w_pool,
                 "twin_noise_band_95_on_mean": boot(d8)["ci95"],
                 "note": "the two runs differ in engine (vllm vs vllm_bf16); tau2 carries "
                         "engine plus run differences"},
        "split_half_st2qp_k4_vs_k4": boot(half),
        "truncation_rate_st2qp_none": trunc, "parse_rate_st2qp_none": parse,
        "components": sets,
        "second_derivation_H1_held": emp,
        "engaged_ids": sorted(engaged), "_tau2": tau2,
    }


# --------------------------------------------------------------------------------------------
# 2. operator (known-gap) cell on Qwen3.8-27B: dose 0 letter reads
# --------------------------------------------------------------------------------------------


def letter_cell(rows: list[dict]) -> dict[str, dict]:
    acc: dict[str, dict] = {}
    for r in rows:
        ls = sorted(r["option_logps"])
        m = np.exp([r["option_logps"][x] for x in ls])
        m = m / m.sum()
        a = acc.setdefault(r["scenario_id"], {"mass": [], "p": []})
        a["mass"].append(float(r["option_mass"]))
        a["p"].append(float(sum(q for x, q in zip(ls, m)
                                if r["norm_status_by_letter"][x] == "violating")))
    return {s: {"mass": float(np.mean(v["mass"])), "p": float(np.mean(v["p"]))}
            for s, v in acc.items()}


def known_gap() -> dict:
    base = Q / "t10b/qwen38_27b/panel"
    cells = {c: letter_cell(jsonl(base / f"{c}.jsonl")) for c in
             ("dl_chat_known_gap", "dl_chat_neutral", "jl_chat_neutral",
              "dl_chat_neutral_pressure_removed")}
    KG, D, J, DN = (cells[c] for c in ("dl_chat_known_gap", "dl_chat_neutral",
                                         "jl_chat_neutral", "dl_chat_neutral_pressure_removed"))

    def contrast(X, Y):
        ids = sorted(s for s in X if s in Y and min(X[s]["mass"], Y[s]["mass"]) >= 0.5)
        dd = [X[s]["p"] - Y[s]["p"] for s in ids]
        return {**boot(dd), "sd_per_scenario": float(np.std(dd, ddof=1))}

    half_kg = letter_cell(jsonl(Q / "st2qp/qwen38_27b/panel/none/dl_chat_known_gap.jsonl"))
    tw = [KG[s]["p"] - half_kg[s]["p"] for s in half_kg if s in KG]
    sol = jload(ST / "data/frontier_panel_gpt61sol_v1.json")["positive_control"]["g_band"]
    return {
        "regime": "dose 0 (enable_thinking_false), letter readout, 8 option orders, 586 panel "
                  "scenarios (t10b screen); thinking-on S(operator) on Qwen is UNGROUNDED",
        "p_violating_mean": {c: float(np.mean([v["p"] for v in x.values()]))
                             for c, x in cells.items()},
        "g_band_KG_minus_J": contrast(KG, J),
        "operator_minus_user_KG_minus_D": contrast(KG, D),
        "operator_minus_plain_KG_minus_Dpr": contrast(KG, DN),
        "user_minus_plain_D_minus_Dpr": contrast(D, DN),
        "twin_KG_t10b_minus_st2qp_half": {**boot(tw),
                                          "note": "same model, dose-0 logprob reads on two pods"},
        "frontier_reference_sol": sol,
    }


# --------------------------------------------------------------------------------------------
# 3. ΔP on H1_held engaged
# --------------------------------------------------------------------------------------------


def delta_p(sp: dict) -> dict:
    tau2 = sp["_tau2"]
    eng = set(sp["engaged_ids"])
    A = by_scenario(jsonl(Q / "st2qp/qwen38_27b/gen/h1_none.jsonl"), eng)
    B = by_scenario(jsonl(Q / "st2q/qwen38_27b/gen/h1_rollouts.jsonl"), eng)
    cq = components(A)
    rtm = boot([np.mean(A[s]) - np.mean(B[s]) for s in sorted(set(A) & set(B))])
    qp = jload(ST / "data/st2qp_read_v1.json")
    qr = jload(ST / "data/st2qr_read_v1.json")
    muse_ids = set(jload(Q / "t16/muse_glimmer_30b/t16_record.json")["phases"]["base_hf"]
                   ["engaged_held"])
    cm = components(by_scenario(jsonl(Q / "t16/muse_glimmer_30b/gen/h1_none.jsonl"), muse_ids))
    t16 = jload(ST / "data/t16_read_v1.json")
    return {
        "qwen_engaged_17": {**cq, "mde_closed_null_k8": mde_closed(17, 8, 1.0, cq["w"], tau2),
                            "mde_closed_null_k8_no_run_term": mde_closed(17, 8, 1.0, cq["w"], 0),
                            "regression_to_mean_P_re_minus_P_sel": rtm,
                            "recorded_mde_care_dpo_q": qp["arms"]["care"]["dP_record"]["mde"],
                            "recorded_mde_care_dpo_q2": qr["care2"]["dP_record"]["mde"]},
        "muse_engaged_54": {**cm, "mde_closed_null_k8": mde_closed(54, 8, 1.0, cm["w"], 0),
                            "recorded_mde_care_dpo": t16["arms"]["care"]["dP_record"]["mde"]},
    }


# --------------------------------------------------------------------------------------------
# 4. MoReBench response field at matched length (MRB v0.9.3 estimator)
# --------------------------------------------------------------------------------------------


def mrb() -> dict:
    sys.path.insert(0, str(REPO / "papers/mrb_morebench/scripts"))
    sys.path.insert(0, str(REPO))
    import mrb_g1_analyze as A  # noqa: E402
    import mrb_p2_length as L  # noqa: E402

    from deepsteer.datasets.morebench import load_morebench  # noqa: E402

    out = REPO / "papers/mrb_morebench/outputs/p2"
    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    us = A.unit_scores(A.read_jsonl(src(out / "judge/shard0.jsonl")), weights)
    gens = {}
    for f in sorted((out / "gen").glob("*.jsonl")):
        gens.update({r["job_id"]: r for r in A.read_jsonl(src(f))})
    sids = [it.scenario_id for it in items]
    rng = np.random.default_rng(SEED)
    res = {"estimator": "MRB_PLAN v0.9.3 Δ_adj = Δ − β_none × Δlength (mrb_p2_length.estimates)",
           "k": 2, "bar_of_record": -0.87}
    for m in L.MODELS:
        d = L.arrays(us, gens, m, sids)
        full = [L.estimates(d, rng.integers(0, 500, 500))["delta_adj"] for _ in range(1000)]
        se200, pt200 = [], []
        for _ in range(20):
            sub = rng.choice(500, 200, replace=False)
            bs = [L.estimates(d, sub[rng.integers(0, 200, 200)])["delta_adj"] for _ in range(400)]
            se200.append(float(np.std(bs, ddof=1)))
            pt200.append(L.estimates(d, sub)["delta_adj"])
        # matched null from none alone: k0 vs k1 of none (one rollout per arm, raw score)
        s0 = d["none"][0]
        nd = s0[:, 1] - s0[:, 0]
        se_null_200_k1 = float(np.std(nd, ddof=1) / sqrt(200))
        se = float(np.mean(se200))
        res[m] = {"se_delta_adj_500": float(np.std(full, ddof=1)),
                  "mde_delta_adj_500": float(Z * np.std(full, ddof=1)),
                  "se_delta_adj_200": se, "mde_delta_adj_200": Z * se,
                  "delta_adj_200_subsets_range": [float(min(pt200)), float(max(pt200))],
                  "null_none_k0_vs_k1_raw_se_200_one_rollout_each": se_null_200_k1,
                  "null_none_k0_vs_k1_raw_mde_200_k2_scaled": Z * se_null_200_k1 / sqrt(2),
                  "fail_detected_80pct_if_true_cost_below": -0.87 - Z * se,
                  "beat_claim_80pct_needs_true_cost_above": -0.87 + Z * se}
    return res


# --------------------------------------------------------------------------------------------
# 5. MACHIAVELLI (MCH rows, read as recorded) and 6. capability
# --------------------------------------------------------------------------------------------


def mch() -> dict:
    d = jload(REPO / "papers/mch_machiavelli/outputs/mch_newmodel_rows.json")
    h2 = d["rows"]["qwen3_30b_instruct"]["H2"]
    g2 = d["rows"]["gpt_oss_20b"]["H2"]
    pick = ("mde_k2_registered_formula", "mde_k2_measured_paired_se", "se_analytic", "point")
    return {"qwen3_30b_a3b_instruct_2507": {m: {x: h2[m][x] for x in pick}
                                            for m in ("violations", "reward")},
            "gpt_oss_20b": {m: {x: g2[m][x] for x in pick} for m in ("violations", "reward")},
            "note": "MCH's Qwen row is Qwen3-30B-A3B-Instruct-2507 (MoE, non-thinking), not the "
                    "CST subject Qwen3.8-27B; k = 2, 30 test games, 1,024 tokens"}


def capability() -> dict:
    def acc(p):
        return {(r["dataset"], str(r["index"])): float(str(r["correct"]) == "True")
                for r in jsonl(p)}
    n0 = acc(Q / "st2qp/qwen38_27b/cap/none.jsonl")
    out = {}
    for arm, p in (("care", Q / "st2qp/qwen38_27b/cap/care.jsonl"),
                   ("base", Q / "st2qp/qwen38_27b/cap/base.jsonl"),
                   ("care2", Q / "st2qr/qwen38_27b/cap/care.jsonl")):
        a = acc(p)
        ks = sorted(set(a) & set(n0))
        d = np.array([a[x] - n0[x] for x in ks])
        disc = float(np.mean(d != 0))
        se = float(sqrt((disc - d.mean() ** 2) / len(d)))
        out[arm] = {"n": len(ks), "d_accuracy": float(d.mean()), "discordant_share": disc,
                    "se_paired": se, "mde_paired": Z * se}
    out["accuracy_none"] = float(np.mean(list(n0.values())))
    out["bar"] = "point estimate ≥ −0.02 (ST A15.CAP_BAR); not a CI rule"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp = s_proxy()
    c = sp["components"]
    kg = known_gap()
    # MDE grids for ΔS (paired, null variance)
    tau2 = sp["_tau2"]
    grids = {
        "H1_held_basis": mde_grid({"n586_panel": 586, "n132_H1_held": 132},
                                  c["H1_held_132"]["w"], tau2),
        "H1_engaged_basis": mde_grid({"n17_engaged": 17}, c["H1_held_engaged_17"]["w"], tau2),
        "cellA2_forced_basis": mde_grid({"n102_forced": 102}, c["cellA2_forced_102"]["w"], tau2),
        "panel_rule_right_basis": mde_grid({"n586_panel": 586, "n60": 60},
                                           c["panel_rule_right_60"]["w"], 0.0),
        "worst_case_w_0.25": mde_grid({"n586_panel": 586, "n132_H1_held": 132},
                                      0.25, 0.0),
    }
    res = {
        "rule": "CST_PLAN v0.1 §8 T0.4; zero GPU; seeds fixed",
        "seed": SEED, "n_boot": N_BOOT, "z_80_power": Z,
        "formula": {"closed_form": "MDE = 2.80 sqrt((2 w E[1/m|m>0] + 2 tau2) / (n (1-(1-f)^k)^2)),"
                                   " m ~ Bin(k, f)",
                    "n_effective": "S is conditional on an applied tenet: a fraction f of the n·k "
                                   "rollouts enter it, and scenarios with no applied rollout in "
                                   "either arm drop out"},
        "S_proxy": {k: v for k, v in sp.items() if not k.startswith("_")},
        "S_mde_grids": grids,
        "operator_known_gap": kg,
        "delta_P": delta_p(sp),
        "morebench_200": mrb(),
        "machiavelli_k2": mch(),
        "capability": capability(),
        "sources_sha256": SOURCES,
    }
    p = OUT / "t04_power.json"
    p.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("S_mde_grids",)}, indent=1))
    print("twin", json.dumps(res["S_proxy"]["twin"], indent=1, default=float))
    print("emp", json.dumps(res["S_proxy"]["second_derivation_H1_held"], indent=1))
    print("comp", json.dumps(res["S_proxy"]["components"], indent=1))
    print("split", res["S_proxy"]["split_half_st2qp_k4_vs_k4"])
    print("kg", json.dumps(kg, indent=1, default=float))
    print("dP", json.dumps(res["delta_P"], indent=1, default=float))
    print("mrb", json.dumps(res["morebench_200"], indent=1, default=float))
    print("mch", json.dumps(res["machiavelli_k2"], indent=1))
    print("cap", json.dumps(res["capability"], indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
