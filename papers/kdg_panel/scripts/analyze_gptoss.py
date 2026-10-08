#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC v0.2 analysis (C0-C3 + decision-token PR), committed before the pod runs.

    python3 papers/kdg_panel/scripts/analyze_gptoss.py [--dir outputs/p2c/gpt_oss_20b/gpt_oss_20b]

C0 (spec §3, G-A1, G-A2): per scenario, the forced primary readout's argmax option (renormalised
over displayed letters, mean over the 8 permutations) against the strict majority option of the
completed low-effort rollouts (ties and unparsed completed rollouts count as non-agreement;
scenarios with no completed rollout leave the denominator). Pass iff agreement >= 0.80; Wilson CI,
SE and the near-miss label beside it. More than a quarter of rollouts truncated -> C0 descriptive.
C1: the P1-A9 item 2 screen on GPT-OSS's own cells (``analyze_own_screen.screen``), engagement and
screen rate beside OLMo-3's and Llama's (``analysis_screen_rates.json``).
C2: g_band = p_D(known gap) - p_J(neutral) over union primaries clearing the floor in both cells
(``analyze_phase1_session_c``); validated iff the 95% lower bound >= 0.10. Slot named, not
size-compared across models (spec §7 rival iii).
C3: E_prob on the model-free set (all four cells engaged; number of record) with MDE = 2.8 SE, and
on the own screen with the P1-A13 selection-matched null (secondary).
PR (G-A3): decision-token residuals of ``dl_chat_neutral`` (permutation 0, one row per scenario) at
hidden_states index 13 (block 12 output, W4's L12), raw and standardized, with the m-out-of-n
subsampling CI (m = n/2, 500 draws, basic form, the W4-07 method), printed beside 12.79
(post-std, Tier-1 harness) and 9.40 [9.09, 10.65] (raw, W4 in-format sample).
Branch per spec §6, in order: readout invalid (C0 fail) -> instrument not validated (C2 fail) ->
carries the gap (E CI above 0) / not detected with its bar (CI includes 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

from deepsteer.geometry.participation import participation_ratio  # noqa: E402
from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

DEFAULT_DIR = A.OUT / "p2c" / "gpt_oss_20b" / "gpt_oss_20b"
PR_INDEX = 13  # hidden_states[13] = output of block 12 = W4's L12 (forward-hook convention)
PR_REFERENCES = {
    "tier1_post_std": {"pr": 12.79, "harness": "gptoss_tier1.py, post-standardization"},
    "w4_raw": {"pr": 9.40, "ci95": [9.09, 10.65], "harness": "W4 in-format sample, L12, n = 128"},
}


def _status(scen) -> dict:
    return {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }


def forced_argmax(d: Path, cell: str) -> dict[str, str]:
    """Scenario -> option with the highest renormalised probability, mean over permutations."""
    rows = A._rows(d / f"{cell}.jsonl")
    z = np.load(d / f"{cell}.npz")
    lp, tid = z["logp_decision"].astype(np.float64), z["option_token_ids"]
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for k, r in enumerate(rows):
        letters = sorted(r["order"])
        ids = [int(tid[k]["ABCDE".index(L)]) for L in letters]
        m = np.exp(lp[k, ids])
        m = m / m.sum()
        for L, mm in zip(letters, m):
            acc[r["scenario_id"]][r["order"][L]].append(float(mm))
    return {s: max(o, key=lambda x: np.mean(o[x])) for s, o in acc.items()}


def _majorities(gen: list[dict]) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for r in gen:
        if r["reasoning_trace"] == "completed":
            by[r["scenario_id"]].append(r["option_id"])
    return by


def c0(d: Path) -> dict:
    gen = A._rows(d / "c0_generate_low.jsonl")
    by = _majorities(gen)
    trunc = sum(r["reasoning_trace"] == "truncated" for r in gen) / len(gen)
    out = {"n_rollouts": len(gen), "truncation_rate": trunc, "descriptive": trunc > 0.25,
           "parse_methods": dict(sorted(_count(r["parse_method"] for r in gen).items()))}
    for name in ("primary", "direct_final"):
        F = forced_argmax(d, f"c0_forced_{name}")
        agree = [kh.strict_majority(by[s]) == F[s] for s in sorted(by) if s in F]
        out[name] = kh.c0_verdict(agree)
    P, Q = forced_argmax(d, "c0_forced_primary"), forced_argmax(d, "c0_forced_direct_final")
    out["primary_vs_direct_final_argmax_agreement"] = float(np.mean([P[s] == Q[s] for s in P]))
    # G-A7: the T=1.0 discriminating batch, present only when the T=0.7 result was a near-miss
    t1 = d / "c0_generate_low_t1.jsonl"
    if t1.exists():
        by1 = _majorities(A._rows(t1))
        v1 = kh.c0_verdict([kh.strict_majority(by1[s]) == P[s] for s in sorted(by1) if s in P])
        same = v1["pass"] == out["primary"]["pass"]
        out["t1"] = {
            **v1,
            "label": "temperature_robust" if same else "temperature_dependent",
            "note": "verdict of record is the T=0.7 rule; a temperature_dependent result is "
                    "stated at both temperatures in every sentence that uses C0",
        }
    return out


def _count(xs) -> dict:
    c: dict = defaultdict(int)
    for x in xs:
        c[x] += 1
    return c


def subsample_ci(X: np.ndarray, draws: int = 500, seed: int = 0) -> dict:
    """m-out-of-n subsampling CI for the PR, m = n/2, basic form, deviations rescaled by sqrt(m/n).
    Reproduces W4's GPT-OSS [9.09, 10.65] as [8.98, 10.57] on the saved W4 sample (draws differ)."""
    n = X.shape[0]
    m = n // 2
    rng = np.random.default_rng(seed)
    th = participation_ratio(X)
    dev = np.array([participation_ratio(X[rng.choice(n, m, replace=False)]) - th
                    for _ in range(draws)])
    s = np.sqrt(m / n)
    return {"ci95": [float(th - s * np.percentile(dev, 97.5)),
                     float(th - s * np.percentile(dev, 2.5))],
            "method": "m-out-of-n subsampling, basic interval", "m": m, "draws": draws,
            "seed": seed, "rng": "numpy default_rng(seed).choice(n, m, replace=False) per draw"}


def decision_pr(d: Path, index: int = PR_INDEX) -> dict:
    rows = A._rows(d / "dl_chat_neutral.jsonl")
    R = np.load(d / "dl_chat_neutral.npz")["resid_decision"]
    keep = [k for k, r in enumerate(rows) if r["seed"] == 0]
    X = R[keep, index, :].astype(np.float64)
    sd = X.std(0)
    Xs = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
    rec = {"hidden_states_index": index, "n": len(keep), "references": PR_REFERENCES}
    for name, Y in (("raw", X), ("standardized", Xs)):
        sub = subsample_ci(Y)
        rec[name] = {"pr": participation_ratio(Y), "subsampling_ci95": sub["ci95"],
                     "subsampling": sub}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss.json")
    ap.add_argument("--pr-index", type=int, default=PR_INDEX, help="stub tests only")
    a = ap.parse_args(argv)
    d = a.dir
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = _status(scen)
    rep: dict = {"spec": "KDG_GPTOSS_SPEC.md v0.2", "n_boot": A.N_BOOT, "seed": A.SEED,
                 "floor": A.FLOOR, "dir": str(d)}

    rep["c0"] = c0(d)

    Dm = A.option_cell([d], "dl_chat_neutral", status)
    Jm = A.option_cell([d], "jl_chat_neutral", status)
    eng = [
        s for s in status
        if s in Dm and s in Jm and min(Dm[s]["mass"], Jm[s]["mass"]) >= A.FLOOR
    ]
    scr = OS.screen(d, status)
    ref = json.loads((A.DATA / "analysis_screen_rates.json").read_text())["models"]
    rep["c1"] = {
        "n_union": len(status), "n_engaged": len(eng), "engagement_rate": len(eng) / len(status),
        "n_screened": len(scr), "screen_rate_of_engaged": len(scr) / max(len(eng), 1),
        "beside": {k: ref[k] for k in ("olmo3_final", "llama31_instruct_meta") if k in ref},
    }

    K = A.option_cell([d], "dl_chat_known_gap", status)
    ids = sorted(s for s in K if s in Jm and min(K[s]["mass"], Jm[s]["mass"]) >= A.FLOOR)
    g = A.boot(np.array([K[s]["p"] - Jm[s]["p"] for s in ids]))
    lo = g["ci95"][0]
    rep["c2"] = {
        "g_band": g, "p_D_known_gap": A.boot(np.array([K[s]["p"] for s in ids])),
        "verdict": "validated" if (lo is not None and lo >= 0.10) else "not_validated",
        "slot": "harmony developer turn beside the model's own system message (OLMo-3: the "
                "operator prompt replaces the default system prompt); not size-compared",
    }

    T = A.four_cells([d], OS.CHAT, status)
    free = sorted(s for s in T if T[s]["mass_min"] >= A.FLOOR)
    E = A.boot(np.array([A.scales(T[s])["E_prob"] for s in free]))
    se = (E["ci95"][1] - E["ci95"][0]) / (2 * 1.96)
    own = [s for s in scr if s in T and T[s]["mass_min"] >= A.FLOOR]
    tw = [s for s in OS.twin_screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
    e_own = np.array([A.scales(T[s])["E_prob"] for s in own])
    e_rev = np.array([-A.scales(T[s])["E_prob"] for s in tw])
    sel = None
    if len(e_own) and len(e_rev):
        rng = np.random.default_rng(A.SEED)
        diff = (e_own[rng.integers(0, len(e_own), (A.N_BOOT, len(e_own)))].mean(1)
                - e_rev[rng.integers(0, len(e_rev), (A.N_BOOT, len(e_rev)))].mean(1))
        sel = {"E_sel": float(e_own.mean() - e_rev.mean()),
               "ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
               "n_twin_screen": len(tw)}
    rep["c3"] = {
        "model_free": {"n": len(free), "E_prob": E, "mde": 2.8 * se},
        "own_screen": {"n": len(own), "E_prob": A.boot(e_own) if len(e_own) else None,
                       "selection_null": sel},
    }

    rep["decision_token_pr"] = decision_pr(d, a.pr_index)

    c0p = rep["c0"]["primary"]
    if not c0p["pass"] and not rep["c0"]["descriptive"]:
        branch = "readout_invalid"
    elif rep["c2"]["verdict"] != "validated":
        branch = "instrument_not_validated"
    elif E["ci95"][0] > 0:
        branch = "carries_gap"
    elif E["ci95"][1] < 0:
        branch = "negative_excess_unregistered"
    else:
        branch = "not_detected"
    rep["branch"] = branch
    rep["branch_notes"] = {
        "c0_descriptive": rep["c0"]["descriptive"],
        "c0_near_miss": c0p["near_miss"],
        "c0_t1_label": rep["c0"].get("t1", {}).get("label"),
        "wording_not_detected": f"no pressure-attributable gap detectable above "
                                f"{2.8 * se:.3f} on GPT-OSS-20B" if branch == "not_detected"
        else None,
    }
    a.out.write_text(json.dumps(rep, indent=1, default=float))
    f = lambda x: "n/a" if x["mean"] is None else "%.3f [%.3f, %.3f] n%d" % (  # noqa: E731
        x["mean"], *x["ci95"], x["n"])
    print(f"C0 primary: {c0p['agreement']:.3f} Wilson {c0p['wilson_ci95']} -> {c0p['verdict']}; "
          f"truncated {rep['c0']['truncation_rate']:.2%}")
    if "t1" in rep["c0"]:
        t1 = rep["c0"]["t1"]
        print(f"C0 at T=1.0 (G-A7): {t1['agreement']:.3f} -> {t1['verdict']}; {t1['label']}")
    print(f"C1: engaged {len(eng)}/{len(status)}, screened {len(scr)}")
    print(f"C2: g_band {f(g)} -> {rep['c2']['verdict']}")
    print(f"C3: E model-free {f(E)}  MDE {2.8 * se:.4f}")
    pr = rep["decision_token_pr"]
    print(f"PR (hs[{a.pr_index}]): raw {pr['raw']['pr']:.2f} {pr['raw']['subsampling_ci95']}, "
          f"std {pr['standardized']['pr']:.2f}; refs 12.79 post-std (Tier-1), 9.40 raw (W4)")
    print(f"branch: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
