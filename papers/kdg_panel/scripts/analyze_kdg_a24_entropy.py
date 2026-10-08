#!/usr/bin/env python3
"""ANOMALIES KDG-A24 (rule pushed bf74c88 before this ran): does the carrier vs non-carrier
contrast in E survive conditioning on dose-0 letter entropy?

    python3 papers/kdg_panel/scripts/analyze_kdg_a24_entropy.py

Per scenario: H_s = mean entropy (nats) of the dose-0 letter distribution over the four C3 cells
and their permutations; E_s the probability-scale excess; the model-free set per model (all four
cells at option mass >= 0.5). (i) Within-model slope of E_s on H_s, bootstrap CI. (ii) Model
contrasts before and after entropy matching: each model's scenarios reweighted to the pooled H_s
distribution over 10 pooled quantile bins, using only bins where both models of a contrast have
scenarios; bootstrap over scenarios within model; 10,000 draws, seed 0. Outcome over the contrasts
whose unadjusted CI excludes 0: survives (every adjusted CI excludes 0), explained (every adjusted
CI includes 0), partial otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

DIRS = {
    "olmo3_final": SR.MODELS["olmo3_final"],
    "llama31_instruct_meta": SR.MODELS["llama31_instruct_meta"],
    "tulu3_final": SR.MODELS["tulu3_final"],
    "qwen25_instruct": SR.MODELS["qwen25_instruct"],
    "gpt_oss_20b": G.DEFAULT_DIR,
}
CONTRASTS = [("olmo3_final", "tulu3_final"), ("llama31_instruct_meta", "tulu3_final"),
             ("olmo3_final", "qwen25_instruct"), ("llama31_instruct_meta", "qwen25_instruct"),
             ("olmo3_final", "gpt_oss_20b"), ("llama31_instruct_meta", "gpt_oss_20b")]
BINS = 10


def entropy(d: Path, status: dict) -> dict[str, float]:
    acc: dict[str, list] = {}
    for c in OS.CHAT:
        for r in A._rows(d / f"{c}.jsonl"):
            if r["scenario_id"] not in status:
                continue
            lp = np.array(list(r["option_logps"].values()), float)
            p = np.exp(lp - lp.max())
            p /= p.sum()
            acc.setdefault(r["scenario_id"], []).append(float(-(p * np.log(p + 1e-300)).sum()))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def ci(v) -> list[float]:
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    data = {}
    for k, d in DIRS.items():
        T = A.four_cells([d], OS.CHAT, status)
        H = entropy(d, status)
        ids = sorted(s for s in T if T[s]["mass_min"] >= A.FLOOR and s in H)
        data[k] = (np.array([A.scales(T[s])["E_prob"] for s in ids]), np.array([H[s] for s in ids]))
    edges = np.quantile(np.concatenate([h for _, h in data.values()]), np.linspace(0, 1, BINS + 1))
    edges[-1] += 1e-9
    pooled = np.histogram(np.concatenate([h for _, h in data.values()]), edges)[0].astype(float)
    rep: dict = {"rule": "ANOMALIES KDG-A24 (bf74c88)", "n_boot": A.N_BOOT, "seed": A.SEED,
                 "bins": BINS, "models": {}, "contrasts": {}}
    rng = np.random.default_rng(A.SEED)
    for k, (E, H) in data.items():
        X = np.c_[np.ones_like(H), H]
        beta = np.linalg.lstsq(X, E, rcond=None)[0][1]
        bs = []
        for _ in range(A.N_BOOT):
            i = rng.integers(0, len(E), len(E))
            bs.append(np.linalg.lstsq(X[i], E[i], rcond=None)[0][1])
        rep["models"][k] = {"n": len(E), "E_mean": float(E.mean()), "H_mean": float(H.mean()),
                            "slope_E_on_H": float(beta), "slope_ci95": ci(bs)}

    def adj_mean(E, H, keep):
        b = np.digitize(H, edges[1:-1])
        w = np.zeros_like(E)
        tot = pooled[keep].sum()
        for j in keep:
            m = b == j
            if m.any():
                w[m] = (pooled[j] / tot) / m.sum()
        return float((w * E).sum())

    for a, b in CONTRASTS:
        (Ea, Ha), (Eb, Hb) = data[a], data[b]
        ba, bb = np.digitize(Ha, edges[1:-1]), np.digitize(Hb, edges[1:-1])
        keep = [j for j in range(BINS) if (ba == j).any() and (bb == j).any()]
        rng = np.random.default_rng(A.SEED)
        un, ad = [], []
        for _ in range(A.N_BOOT):
            ia, ib = rng.integers(0, len(Ea), len(Ea)), rng.integers(0, len(Eb), len(Eb))
            un.append(Ea[ia].mean() - Eb[ib].mean())
            kb = [j for j in keep if (np.digitize(Ha[ia], edges[1:-1]) == j).any()
                  and (np.digitize(Hb[ib], edges[1:-1]) == j).any()]
            ad.append(adj_mean(Ea[ia], Ha[ia], kb) - adj_mean(Eb[ib], Hb[ib], kb))
        rep["contrasts"][f"{a} - {b}"] = {
            "unadjusted": float(Ea.mean() - Eb.mean()), "unadjusted_ci95": ci(un),
            "adjusted": adj_mean(Ea, Ha, keep) - adj_mean(Eb, Hb, keep), "adjusted_ci95": ci(ad),
            "bins_common": len(keep)}
    sig = [v for v in rep["contrasts"].values()
           if v["unadjusted_ci95"][0] > 0 or v["unadjusted_ci95"][1] < 0]
    still = [v["adjusted_ci95"][0] > 0 or v["adjusted_ci95"][1] < 0 for v in sig]
    rep["outcome"] = ("survives" if sig and all(still) else "explained" if sig and not any(still)
                      else "partial" if sig else "no_unadjusted_contrast")
    (A.DATA / "analysis_kdg_a24_entropy.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["models"].items():
        print(f"{k:22s} n {v['n']:3d} E {v['E_mean']:+.4f} H {v['H_mean']:.3f} slope "
              f"{v['slope_E_on_H']:+.4f} [{v['slope_ci95'][0]:+.4f}, {v['slope_ci95'][1]:+.4f}]")
    for k, v in rep["contrasts"].items():
        u, a_ = v["unadjusted_ci95"], v["adjusted_ci95"]
        print(f"{k:44s} unadj {v['unadjusted']:+.4f} [{u[0]:+.4f}, {u[1]:+.4f}]  adj "
              f"{v['adjusted']:+.4f} [{a_[0]:+.4f}, {a_[1]:+.4f}] ({v['bins_common']} bins)")
    print("outcome:", rep["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
