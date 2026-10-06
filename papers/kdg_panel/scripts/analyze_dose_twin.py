#!/usr/bin/env python3
"""P1-A12 (KDG_PHASE1_SPEC, pushed c263eb2; GPU-1, author go 2026-10-05), committed before the data.

    python3 papers/kdg_panel/scripts/analyze_dose_twin.py

Per model, on the dose set of record (screened scenarios with every forced cell defined), paired by
scenario: Δ_P = p_D(reasoning) − p_D(filler) on primaries (the result of record), Δ_T the same on
the pressure-removed twins, ΔE_delib = Δ_P − Δ_T; the truncated-filler (own TF) versions beside.
Bootstrap 10,000, seed 0; bar = 2.8 × SE of ΔE_delib. Branches (written before data):
(a) "reduces_pressure_part" iff ΔE_delib's CI lies entirely below 0; (b) "removes_at_rest_asymmetry"
iff ΔE_delib's CI includes 0 and Δ_T's CI lies entirely below 0; "unresolved" otherwise. Per model;
no cross-model size comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

P1A, P1B, P1C, P1D = (A.OUT / p for p in ("p1a", "p1b", "p1c", "p1d"))
SUFFIX = "_pressure_removed"
# primaries: (forced reasoning + filler dir, own-TF dir); twins: one dir; screen of record
MODELS = {
    "olmo3_instruct": {
        "prim": P1A / "final_dose_bf" / "olmo3_instruct",
        "prim_tf": P1B / "dose_controls" / "olmo3_instruct",
        "twin": P1D / "olmo_dose_twin" / "olmo3_instruct",
        "screen": "screened_ids_a17_union.json",
    },
    "llama31_instruct_meta": {
        "prim": P1C / "llama_dose" / "llama31_instruct_meta",
        "prim_tf": P1C / "llama_dose" / "llama31_instruct_meta",
        "twin": P1D / "llama_dose_twin" / "llama31_instruct_meta",
        "screen": "screened_ids_llama31_meta.json",
    },
}


def branch(dE: dict, dT: dict) -> str:
    """The P1-A12 rule on two bootstrap summaries (``ci95`` = [lo, hi])."""
    lo, hi = dE["ci95"]
    if hi < 0:
        return "reduces_pressure_part"
    if lo <= 0 <= hi and dT["ci95"][1] < 0:
        return "removes_at_rest_asymmetry"
    return "unresolved"


def contrast(P: dict, T: dict, ids: list[str], boot=A.boot) -> dict:
    """P, T: {"D2": {sid: p}, "F": {...}, "TF": {...}} for primaries and twins."""
    out: dict = {"n": len(ids)}
    for ref in ("F", "TF"):
        dP = np.array([P["D2"][s] - P[ref][s] for s in ids])
        dT = np.array([T["D2"][s] - T[ref][s] for s in ids])
        bP, bT, bE = boot(dP), boot(dT), boot(dP - dT)
        se = (bE["ci95"][1] - bE["ci95"][0]) / (2 * 1.96)
        out[f"vs_{ref}"] = {
            "delta_P": bP,
            "delta_T": bT,
            "delta_E_delib": bE,
            "bar": 2.8 * se,
            "branch": branch(bE, bT),
        }
    out["verdict"] = out["vs_F"]["branch"]  # filler is the pre-registered reference; TF beside
    return out


def ratio_fork(P: dict, T: dict, ids: list[str], seed: int = A.SEED) -> dict:
    """P1-A14 (post-hoc fork, pushed before this ran): L = log R_P − log R_T, R = mean D2 / mean
    ref, paired bootstrap over scenarios; ΔE predicted by a common ratio beside the observed ΔE."""
    out: dict = {}
    n = len(ids)
    idx = np.random.default_rng(seed).integers(0, n, size=(A.N_BOOT, n))
    for ref in ("F", "TF"):
        dp, rp = (np.array([P[k][s] for s in ids]) for k in ("D2", ref))
        dt, rt = (np.array([T[k][s] for s in ids]) for k in ("D2", ref))
        L = float(np.log(dp.mean() / rp.mean()) - np.log(dt.mean() / rt.mean()))
        Lb = np.log(dp[idx].mean(1) / rp[idx].mean(1)) - np.log(dt[idx].mean(1) / rt[idx].mean(1))
        lo, hi = (float(v) for v in np.percentile(Lb, [2.5, 97.5]))
        rbar = (dp.sum() + dt.sum()) / (rp.sum() + rt.sum())
        out[f"vs_{ref}"] = {
            "R_P": float(dp.mean() / rp.mean()),
            "R_T": float(dt.mean() / rt.mean()),
            "L": {"mean": L, "ci95": [lo, hi], "n": n, "point_in_ci": lo <= L <= hi},
            "R_common": float(rbar),
            "delta_E_predicted_common_ratio": float((rbar - 1) * (rp.mean() - rt.mean())),
            "delta_E_observed": float((dp - rp).mean() - (dt - rt).mean()),
            "branch": (
                "pressure_specific_beyond_proportional"
                if hi < 0
                else ("twin_heavier" if lo > 0 else "proportional")
            ),
        }
    out["verdict"] = out["vs_F"]["branch"]
    return out


def per_scen(d: Path, cell: str, status, ids_chat, ids_sp) -> dict[str, float]:
    acc: dict[str, list] = {}
    for sid, m, _ in A.row_masses(d, cell, status, ids_chat, ids_sp):
        acc.setdefault(sid, []).append(m)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep: dict = {
        "amendment": "P1-A12 (c263eb2); P1-A14 ratio fork",
        "n_boot": A.N_BOOT,
        "seed": A.SEED,
        "models": {},
    }
    for key, m in MODELS.items():
        if not (m["twin"] / f"d_chat_dose2_bf{SUFFIX}_forced.jsonl").exists():
            print(f"{key}: twin cells not present, skipped")
            continue
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ids_chat, ids_sp = C.llama_letter_ids(m["prim"])
        else:
            import analyze_continuous as ac

            ids_chat, ids_sp = ac._tok_ids()
        P = {
            "D2": per_scen(m["prim"], "d_chat_dose2_bf_forced", status, ids_chat, ids_sp),
            "F": per_scen(m["prim"], "d_chat_dose2_filler_bf_forced", status, ids_chat, ids_sp),
            "TF": per_scen(m["prim_tf"], "d_chat_dose2_filler_tf_forced", status, ids_chat, ids_sp),
        }
        T = {
            "D2": per_scen(m["twin"], f"d_chat_dose2_bf{SUFFIX}_forced", status, ids_chat, ids_sp),
            "F": per_scen(
                m["twin"], f"d_chat_dose2_filler_bf{SUFFIX}_forced", status, ids_chat, ids_sp
            ),
            "TF": per_scen(
                m["twin"], f"d_chat_dose2_filler_tf{SUFFIX}_forced", status, ids_chat, ids_sp
            ),
        }
        screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        ids = sorted(s for s in screen if all(s in X for X in (*P.values(), *T.values())))
        rep["models"][key] = contrast(P, T, ids)
        rep["models"][key]["ratio_fork_P1_A14"] = ratio_fork(P, T, ids)
        r = rep["models"][key]
        for ref in ("F", "TF"):
            v = r["ratio_fork_P1_A14"][f"vs_{ref}"]
            print(
                f"{key:22s} P1-A14 vs {ref:2s} R_P {v['R_P']:.3f} R_T {v['R_T']:.3f} L "
                f"{v['L']['mean']:+.3f} [{v['L']['ci95'][0]:+.3f}, {v['L']['ci95'][1]:+.3f}] "
                f"dE pred {v['delta_E_predicted_common_ratio']:+.4f} obs "
                f"{v['delta_E_observed']:+.4f} -> {v['branch']}"
            )
        for ref in ("F", "TF"):
            v = r[f"vs_{ref}"]
            print(
                f"{key:22s} vs {ref:2s} n {r['n']:3d}  dP {v['delta_P']['mean']:+.4f} "
                f"dT {v['delta_T']['mean']:+.4f} [{v['delta_T']['ci95'][0]:+.4f}, "
                f"{v['delta_T']['ci95'][1]:+.4f}]  dE {v['delta_E_delib']['mean']:+.4f} "
                f"[{v['delta_E_delib']['ci95'][0]:+.4f}, {v['delta_E_delib']['ci95'][1]:+.4f}] "
                f"bar {v['bar']:.4f} -> {v['branch']}"
            )
    (A.DATA / "analysis_dose_twin.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
