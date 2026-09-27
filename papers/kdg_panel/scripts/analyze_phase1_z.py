#!/usr/bin/env python3
"""Phase 1 zero-GPU checks Z1a, Z1b, Z2 (KDG_PHASE1_SPEC.md v0.2 §2, pushed 090b55a before this ran).

    python3 papers/kdg_panel/scripts/analyze_phase1_z.py

Inputs: the A17 union raw cells (KDG-2 + KDG-3 ``*_raw*.jsonl`` for OLMo-3 base and instruct), the
same per-permutation option log-probs ``analyze_paper8.raw_cell`` reads, and the same shared-floor
subset (both models above the 0.5 floor on the scenario and its pressure-removed twin: 192).

Z1a  log-odds recompute (baseline compression): E_logit per scenario and model, paired Δ_logit;
     acting side S_act and judging side S_judge on the same scale.
Z1b  sharpening: (i) judging-side prediction D_judge = S_judge,inst − k_act·S_judge,base with k_act
     re-estimated in every bootstrap draw; (ii, primary) per-model scale σ_s = std of the
     mean-centred option log-probs (renormalised over displayed letters) on the pressure-removed
     twins, averaged over permutations and over the two frames; Ẽ_s = E_logit,s / σ_s; paired Δ̃.
Z2   KDG-A6 option-mass split of the instruct twins' g_null (median split; 0.5–0.7 vs above).

Bootstrap: 10,000 draws over scenarios, seed 0, percentile 95% intervals (spec §2).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_pilot as ap  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

N_BOOT = 10_000
SEED = 0
FLOOR = ap.BASE_FLOOR
CLIP = 1e-4
RUNS = [REPO / "papers/kdg_panel/outputs/kdg2", REPO / "papers/kdg_panel/outputs/kdg3"]
DATA = REPO / "papers/kdg_panel/data"
CELLS = ("d_raw", "j_raw", "d_raw_pressure_removed", "j_raw_pressure_removed")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p) - np.log1p(-p)


def read_cell(model: str, cell: str, status: dict) -> dict[str, dict]:
    """scenario -> {mass, p (violating mass renormalised over displayed letters, mean over
    permutations; identical to analyze_paper8.raw_cell), sigma (mean over permutations of the std
    of the mean-centred renormalised option log-probs), entropy (nats, same averaging)}."""
    out: dict[str, dict] = {}
    for run in RUNS:
        path = run / model / f"{cell}.jsonl"
        for sid, rs in ap._by_scenario(ap._rows(path)).items():
            if sid not in status or sid.endswith("S"):
                continue
            pv, sg, en = [], [], []
            for r in rs:
                lp = r["option_logps"]
                letters = sorted(lp)
                raw = np.asarray([lp[L] for L in letters], float)
                m = np.exp(raw)
                m = m / m.sum()
                logm = np.log(m)
                pv.append(sum(mm for L, mm in zip(letters, m)
                              if status[sid].get(r["order"][L]) == "violating"))
                sg.append(float(np.std(logm - logm.mean())))
                en.append(float(-(m * logm).sum()))
            out[sid] = {"mass": float(np.mean([r["option_mass"] for r in rs])), "p": float(np.mean(pv)),
                        "sigma": float(np.mean(sg)), "entropy": float(np.mean(en)), "n": len(rs)}
    return out


def model_table(model: str, status: dict) -> dict[str, dict]:
    c = {k: read_cell(model, k, status) for k in CELLS}
    out = {}
    for sid, D in c["d_raw"].items():
        J, Dn, Jn = c["j_raw"].get(sid), c["d_raw_pressure_removed"].get(sid), c["j_raw_pressure_removed"].get(sid)
        if J is None or Dn is None or Jn is None:
            continue
        out[sid] = {
            "above": min(D["mass"], J["mass"]) >= FLOOR,
            "above_null": min(Dn["mass"], Jn["mass"]) >= FLOOR,
            "twin_mass": min(Dn["mass"], Jn["mass"]),
            "pD": D["p"], "pJ": J["p"], "pDn": Dn["p"], "pJn": Jn["p"],
            "sigma_twin_D": Dn["sigma"], "sigma_twin_J": Jn["sigma"],
            "sigma": 0.5 * (Dn["sigma"] + Jn["sigma"]),
            "entropy_twin": 0.5 * (Dn["entropy"] + Jn["entropy"]),
        }
    return out


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summarize(point: float, draws: np.ndarray) -> dict:
    lo, hi = ci(draws)
    return {"mean": float(point), "ci95": [lo, hi], "point_in_ci": bool(lo <= point <= hi)}


def main() -> int:
    scen, _ = load_scenario_dir(sorted(DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag")}
    B, I = model_table("olmo3_base", status), model_table("olmo3_instruct", status)
    shared = sorted(s for s in B if s in I and B[s]["above"] and I[s]["above"]
                    and B[s]["above_null"] and I[s]["above_null"])
    n = len(shared)
    # "the shared-192 subset is the A17 subset" (same floor, same cells): fail loudly otherwise
    assert n == 192, f"shared subset is {n}, not the A17 192: loader drifted from analyze_paper8"

    def arr(T, k):
        return np.asarray([T[s][k] for s in shared], float)

    pr = {}
    for lab, T in (("base", B), ("instruct", I)):
        pD, pJ, pDn, pJn = (arr(T, k) for k in ("pD", "pJ", "pDn", "pJn"))
        pr[lab] = {
            "E_prob": (pD - pJ) - (pDn - pJn),
            "E_logit": (logit(pD) - logit(pJ)) - (logit(pDn) - logit(pJn)),
            "S_act": logit(pD) - logit(pDn),
            "S_judge": logit(pJ) - logit(pJn),
            "act_prob": pD - pDn, "judge_prob": pJ - pJn,
            "sigma": arr(T, "sigma"), "sigma_D": arr(T, "sigma_twin_D"), "sigma_J": arr(T, "sigma_twin_J"),
            "entropy": arr(T, "entropy_twin"),
        }
    for lab in pr:
        pr[lab]["E_norm"] = pr[lab]["E_logit"] / pr[lab]["sigma"]

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    b, i = pr["base"], pr["instruct"]

    def paired(key):
        d = i[key] - b[key]
        return {"base": summarize(b[key].mean(), b[key][idx].mean(1)),
                "instruct": summarize(i[key].mean(), i[key][idx].mean(1)),
                "diff_instruct_minus_base": summarize(d.mean(), d[idx].mean(1))}

    rep: dict = {
        "spec": "KDG_PHASE1_SPEC.md v0.2 §2 (commit 090b55a, pushed before computation)",
        "n_shared": n, "n_boot": N_BOOT, "seed": SEED, "clip": CLIP, "floor": FLOOR,
        "second_derivation_prob_scale": paired("E_prob"),
        "second_derivation_acting_prob": paired("act_prob"),
        "second_derivation_judging_prob": paired("judge_prob"),
    }

    # ---- Z1a
    z1a = {"E_logit": paired("E_logit"), "S_act": paired("S_act"), "S_judge": paired("S_judge")}
    dprob = rep["second_derivation_prob_scale"]["diff_instruct_minus_base"]
    dl = z1a["E_logit"]["diff_instruct_minus_base"]
    same_sign = np.sign(dl["mean"]) == np.sign(dprob["mean"])
    excl = dl["ci95"][0] > 0 or dl["ci95"][1] < 0
    z1a["verdict"] = "survives_compression" if (excl and same_sign) else "compression_dependent"
    rep["Z1a"] = z1a

    # ---- Z1b (i): judging-side prediction, k_act re-estimated per draw
    Sab, Sai, Sjb, Sji = b["S_act"], i["S_act"], b["S_judge"], i["S_judge"]
    k_pt = Sai.mean() / Sab.mean()
    dj_pt = Sji.mean() - k_pt * Sjb.mean()
    k_bs = Sai[idx].mean(1) / Sab[idx].mean(1)
    dj_bs = Sji[idx].mean(1) - k_bs * Sjb[idx].mean(1)
    sjb = summarize(Sjb.mean(), Sjb[idx].mean(1))
    uninformative = sjb["ci95"][0] <= 0 <= sjb["ci95"][1]
    if uninformative:
        v1 = "uninformative (S_judge,base CI includes 0)"
    elif ci(dj_bs)[1] < 0:
        v1 = "not_explained_by_sharpening"
    else:
        v1 = "consistent_with_sharpening"
    rep["Z1b_i"] = {"k_act": summarize(k_pt, k_bs), "S_judge_base": sjb,
                    "D_judge": summarize(dj_pt, dj_bs), "verdict": v1}

    # ---- Z1b (ii, primary): twin-measured scale normalization
    ratio = i["sigma"] / b["sigma"]
    k_twin_pt = float(np.median(ratio))
    k_twin_bs = np.median(ratio[idx], axis=1)
    z1b2 = {
        "sigma": paired("sigma"), "sigma_D_twin": paired("sigma_D"), "sigma_J_twin": paired("sigma_J"),
        "entropy_twin": paired("entropy"),
        "k_twin_median_ratio": summarize(k_twin_pt, k_twin_bs),
        "E_norm": paired("E_norm"),
        "min_sigma": {"base": float(b["sigma"].min()), "instruct": float(i["sigma"].min())},
    }
    dn = z1b2["E_norm"]["diff_instruct_minus_base"]
    z1b2["verdict"] = ("widening_survives_sharpening" if dn["ci95"][0] > 0
                       else "sharpening_explained")
    rep["Z1b_ii_primary"] = z1b2

    # ---- Z2: instruct twins' g_null by option mass (the A6 discriminator)
    gnull = (i_pDn := arr(I, "pDn")) - arr(I, "pJn")
    del i_pDn
    mass = arr(I, "twin_mass")
    med = float(np.median(mass))
    low, high = gnull[mass <= med], gnull[mass > med]
    band, above = gnull[mass < 0.7], gnull[mass >= 0.7]

    def unpaired(a, c):
        r = np.random.default_rng(SEED)
        ia = r.integers(0, len(a), size=(N_BOOT, len(a)))
        ic = r.integers(0, len(c), size=(N_BOOT, len(c)))
        return summarize(a.mean() - c.mean(), a[ia].mean(1) - c[ic].mean(1))

    split_med = unpaired(low, high)
    split_band = unpaired(band, above) if len(band) >= 10 else None
    r_b_support = split_med["ci95"][1] < 0
    rep["Z2"] = {
        "mass_definition": "min(D-twin, J-twin) raw option mass, instruct, shared 192",
        "median_mass": med, "n_low": int(len(low)), "n_high": int(len(high)),
        "g_null_low": float(low.mean()), "g_null_high": float(high.mean()),
        "delta_low_minus_high": split_med,
        "n_band_0.5_0.7": int(len(band)), "n_above_0.7": int(len(above)),
        "delta_band_minus_above": split_band,
        "g_null_all": summarize(gnull.mean(), gnull[idx].mean(1)),
        "verdict": ("R_b_supported" if r_b_support else "R_b_loses_cheapest_support"),
    }

    (DATA / "analysis_z1_scale.json").write_text(json.dumps(
        {k: rep[k] for k in rep if k != "Z2"}, indent=1))
    (DATA / "analysis_z2_a6_mass.json").write_text(json.dumps(
        {"spec": rep["spec"], "n_boot": N_BOOT, "seed": SEED, "Z2": rep["Z2"]}, indent=1))
    with open(DATA / "per_scenario_phase1_z.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["E_prob", "E_logit", "S_act", "S_judge", "sigma", "sigma_D", "sigma_J", "entropy", "E_norm"]
        w.writerow(["scenario_id"] + [f"{k}_{m}" for m in ("base", "instruct") for k in keys]
                   + ["instruct_twin_mass", "instruct_g_null"])
        for j, s in enumerate(shared):
            w.writerow([s] + [pr[m][k][j] for m in ("base", "instruct") for k in keys]
                       + [mass[j], gnull[j]])
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
