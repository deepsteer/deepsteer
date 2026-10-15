#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 3: two-way intervals for every dose-arm quantity the v2 paper prints.

    python3 papers/kdg_panel/scripts/analyze_v2_dose_twoway.py

On the dose sets of record (OLMo-3 n 130, Llama-3.1 Meta n 114; as analyze_kdg_a23_twoway), per
scenario and rollout index (16 per cell): the Figure 8 contrasts (reasoning, brief reasoning, the
truncated filler and, on OLMo-3, the norm-named arm, each against the filler of record; reasoning
against the truncated filler), the twins' contrasts and Delta E, the norm-salience share (NS - F) /
(D2 - F), and the P1-A14 log-ratio difference L against both references. Each with the scenario
bootstrap (rollout indices fixed) and the two-way bootstrap (scenarios and rollout indices
resampled independently, one index set for every cell in a draw), 10,000 draws, seed 0. Also the
panel-side arm of the panel-vs-H1 brake difference: the panel Delta_P under the two-way bootstrap
against the H1 Delta_P under the scenario bootstrap (H1 was sampled with per-scenario seeds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_brake_panel_vs_h1 as PH  # noqa: E402
import analyze_dose_twin as T  # noqa: E402
import analyze_h1_carriers as HC  # noqa: E402
import analyze_kdg_a23_twoway as W  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

NAMES = ("d2_minus_f", "d2_minus_tf", "d1_minus_f", "tf_minus_f", "ns_minus_f",
         "twin_d2_minus_f", "twin_d2_minus_tf", "delta_E_vs_f", "delta_E_vs_tf",
         "ns_share", "logratio_vs_f", "logratio_vs_tf")


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    ref = json.loads((A.DATA / "analysis_kdg_a23_twoway.json").read_text())["models"]
    rep: dict = {"rule": "KDG_PHASE1_SPEC P1-A17 item 3", "n_boot": A.N_BOOT, "seed": A.SEED,
                 "models": {}}
    for key, m in T.MODELS.items():
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ic, isp = C.llama_letter_ids(m["prim"])
        else:
            import analyze_continuous as ac

            ic, isp = ac._tok_ids()
        cells = {("P", "D2"): (m["prim"], "d_chat_dose2_bf_forced"),
                 ("P", "D1"): (m["prim"], "d_chat_dose1_bf_forced"),
                 ("P", "F"): (m["prim"], "d_chat_dose2_filler_bf_forced"),
                 ("P", "TF"): (m["prim_tf"], "d_chat_dose2_filler_tf_forced"),
                 ("T", "D2"): (m["twin"], f"d_chat_dose2_bf{W.SUF}_forced"),
                 ("T", "F"): (m["twin"], f"d_chat_dose2_filler_bf{W.SUF}_forced"),
                 ("T", "TF"): (m["twin"], f"d_chat_dose2_filler_tf{W.SUF}_forced")}
        has_ns = (m["prim_tf"] / "d_chat_dose2_filler_ns_forced.jsonl").exists()
        if has_ns:
            cells[("P", "NS")] = (m["prim_tf"], "d_chat_dose2_filler_ns_forced")
        per = {k: T.per_scen(d, c, status, ic, isp) for k, (d, c) in cells.items()}
        screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        base = (("P", "D2"), ("P", "F"), ("P", "TF"), ("T", "D2"), ("T", "F"), ("T", "TF"))
        ids = sorted(s for s in screen if all(s in per[k] for k in base))
        assert len(ids) == ref[key]["n"], (key, len(ids), ref[key]["n"])
        assert all(s in per[k] for k in per for s in ids), (key, "an added cell misses a scenario")
        X = {k: W.matrix(d, c, status, ic, isp, ids) for k, (d, c) in cells.items()}
        n, K = len(ids), W.K

        def stats(si, ri):
            mu = {k: X[k][np.ix_(si, ri)].mean(1) for k in X}
            P = {k[1]: mu[k] for k in mu if k[0] == "P"}
            Tw = {k[1]: mu[k] for k in mu if k[0] == "T"}
            d2f, d2tf = (P["D2"] - P["F"]).mean(), (P["D2"] - P["TF"]).mean()
            tf2f, tf2tf = (Tw["D2"] - Tw["F"]).mean(), (Tw["D2"] - Tw["TF"]).mean()
            nsf = (P["NS"] - P["F"]).mean() if "NS" in P else np.nan
            return np.array([
                d2f, d2tf, (P["D1"] - P["F"]).mean(), (P["TF"] - P["F"]).mean(), nsf,
                tf2f, tf2tf, d2f - tf2f, d2tf - tf2tf, nsf / d2f,
                np.log(P["D2"].mean() / P["F"].mean()) - np.log(Tw["D2"].mean() / Tw["F"].mean()),
                np.log(P["D2"].mean() / P["TF"].mean())
                - np.log(Tw["D2"].mean() / Tw["TF"].mean())])

        point = stats(np.arange(n), np.arange(K))
        rng = np.random.default_rng(A.SEED)
        two = np.array([stats(rng.integers(0, n, n), rng.integers(0, K, K))
                        for _ in range(A.N_BOOT)])
        rng = np.random.default_rng(A.SEED)
        one = np.array([stats(rng.integers(0, n, n), np.arange(K)) for _ in range(A.N_BOOT)])
        out = {"n": n}
        for i, name in enumerate(NAMES):
            if np.isnan(point[i]):
                continue
            c1, c2 = W.ci(one[:, i]), W.ci(two[:, i])
            out[name] = {"point": float(point[i]), "scenario_ci95": c1, "two_way_ci95": c2,
                         "width_ratio": (c2[1] - c2[0]) / (c1[1] - c1[0]),
                         "detectable_2p8se_scenario": float(2.8 * one[:, i].std(ddof=1)),
                         "detectable_2p8se_two_way": float(2.8 * two[:, i].std(ddof=1))}
        # panel-side arm of the panel-vs-H1 brake difference
        hd = A.OUT / "p1e" / HC.MODELS[key] / key
        H, _ = load_scenario_dir([A.DATA / "harm_divergence" / "hdiv_set.json"])
        hids = set(json.loads((A.DATA / "harm_divergence" / "h1_kept_ids.json").read_text())["ids"])
        hstat = {s.id: {o.option_id: o.norm_status for o in s.options} for s in H if s.id in hids}
        h = PH.deltas(hd, hstat, hids, key)
        rng = np.random.default_rng(A.SEED)
        panel_two = two[:, 0]
        hb = h[rng.integers(0, len(h), size=(A.N_BOOT, len(h)))].mean(1)
        diff = panel_two - hb
        out["panel_minus_h1"] = {"point": float(point[0] - h.mean()), "two_way_ci95": W.ci(diff),
                                 "detectable_2p8se": float(2.8 * diff.std(ddof=1)),
                                 "h1_n": len(h)}
        rep["models"][key] = out
        print(key, json.dumps({k: (v if not isinstance(v, dict) else
                                   {kk: (round(vv, 4) if isinstance(vv, float) else
                                         [round(x, 4) for x in vv] if isinstance(vv, list) else vv)
                                    for kk, vv in v.items()}) for k, v in out.items()}))
    (A.DATA / "analysis_v2_dose_twoway.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
