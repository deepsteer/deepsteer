#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A15 (pushed bf74c88 before this ran): two-way bootstrap for the sampled dose arms.

    python3 papers/kdg_panel/scripts/analyze_kdg_a23_twoway.py

On the paired sets of record (analyze_dose_twin: OLMo-3 n 130, Llama-3.1 Meta n 114), per scenario and
rollout index (16 per cell), the forced violating mass. Δ_P, Δ_T and ΔE_delib against the filler and
own-truncated-filler references, with (a) the registered scenario bootstrap and (b) the two-way
bootstrap (scenarios and rollout indices resampled independently; one index set for every cell, so
rollout k stays paired across arms), 10,000 draws, seed 0; plus the delete-one-index jackknife range.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as T  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

K = 16
SUF = "_pressure_removed"


def matrix(d: Path, cell: str, status, ids_chat, ids_sp, ids: list[str]) -> np.ndarray:
    X = np.full((len(ids), K), np.nan)
    pos = {s: i for i, s in enumerate(ids)}
    for sid, m, r in A.row_masses(d, cell, status, ids_chat, ids_sp):
        if sid in pos:
            X[pos[sid], int(r["rollout"])] = m
    assert not np.isnan(X).any(), (cell, "missing rollout")
    return X


def ci(v: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    rep: dict = {"amendment": "P1-A15 (bf74c88)", "n_boot": A.N_BOOT, "seed": A.SEED, "models": {}}
    for key, m in T.MODELS.items():
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ids_chat, ids_sp = C.llama_letter_ids(m["prim"])
        else:
            import analyze_continuous as ac

            ids_chat, ids_sp = ac._tok_ids()
        cells = {
            ("P", "D2"): (m["prim"], "d_chat_dose2_bf_forced"),
            ("P", "F"): (m["prim"], "d_chat_dose2_filler_bf_forced"),
            ("P", "TF"): (m["prim_tf"], "d_chat_dose2_filler_tf_forced"),
            ("T", "D2"): (m["twin"], f"d_chat_dose2_bf{SUF}_forced"),
            ("T", "F"): (m["twin"], f"d_chat_dose2_filler_bf{SUF}_forced"),
            ("T", "TF"): (m["twin"], f"d_chat_dose2_filler_tf{SUF}_forced"),
        }
        per = {k: T.per_scen(d, c, status, ids_chat, ids_sp) for k, (d, c) in cells.items()}
        screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        ids = sorted(s for s in screen if all(s in v for v in per.values()))
        X = {k: matrix(d, c, status, ids_chat, ids_sp, ids) for k, (d, c) in cells.items()}
        n = len(ids)
        out = {"n": n}
        for ref in ("F", "TF"):
            def stats(si, ri):
                mean = {k: X[k][np.ix_(si, ri)].mean(1) for k in X}
                dP = mean[("P", "D2")] - mean[("P", ref)]
                dT = mean[("T", "D2")] - mean[("T", ref)]
                return dP.mean(), dT.mean(), (dP - dT).mean()

            full = np.arange(n), np.arange(K)
            point = stats(*full)
            rng = np.random.default_rng(A.SEED)
            two = np.array([stats(rng.integers(0, n, n), rng.integers(0, K, K))
                            for _ in range(A.N_BOOT)])
            rng = np.random.default_rng(A.SEED)
            one = np.array([stats(rng.integers(0, n, n), np.arange(K)) for _ in range(A.N_BOOT)])
            jack = np.array([stats(np.arange(n), np.delete(np.arange(K), j)) for j in range(K)])
            res = {}
            for i, name in enumerate(("delta_P", "delta_T", "delta_E_delib")):
                c1, c2 = ci(one[:, i]), ci(two[:, i])
                se1 = (c1[1] - c1[0]) / (2 * 1.96)
                res[name] = {
                    "point": float(point[i]),
                    "scenario_ci95": c1, "two_way_ci95": c2,
                    "width_ratio": (c2[1] - c2[0]) / (c1[1] - c1[0]),
                    "jackknife_range": [float(jack[:, i].min()), float(jack[:, i].max())],
                    "jackknife_max_shift_in_se": float(np.abs(jack[:, i] - point[i]).max() / se1),
                }
            res["branch_scenario"] = T.branch(
                {"mean": res["delta_E_delib"]["point"], "ci95": res["delta_E_delib"]["scenario_ci95"]},
                {"mean": res["delta_T"]["point"], "ci95": res["delta_T"]["scenario_ci95"]})
            res["branch_two_way"] = T.branch(
                {"mean": res["delta_E_delib"]["point"], "ci95": res["delta_E_delib"]["two_way_ci95"]},
                {"mean": res["delta_T"]["point"], "ci95": res["delta_T"]["two_way_ci95"]})
            res["delta_P_below_0"] = {"scenario": res["delta_P"]["scenario_ci95"][1] < 0,
                                      "two_way": res["delta_P"]["two_way_ci95"][1] < 0}
            out[f"vs_{ref}"] = res
        rep["models"][key] = out
    (A.DATA / "analysis_kdg_a23_twoway.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["models"].items():
        for ref in ("F", "TF"):
            r = v[f"vs_{ref}"]
            for q in ("delta_P", "delta_T", "delta_E_delib"):
                x = r[q]
                print(f"{k:22s} vs {ref:2s} {q:13s} {x['point']:+.4f} scen "
                      f"[{x['scenario_ci95'][0]:+.4f}, {x['scenario_ci95'][1]:+.4f}] two-way "
                      f"[{x['two_way_ci95'][0]:+.4f}, {x['two_way_ci95'][1]:+.4f}] "
                      f"ratio {x['width_ratio']:.2f} jack {x['jackknife_max_shift_in_se']:.2f} SE")
            print(f"{k:22s} vs {ref:2s} branch scen {r['branch_scenario']} / two-way "
                  f"{r['branch_two_way']}; dP<0 {r['delta_P_below_0']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
