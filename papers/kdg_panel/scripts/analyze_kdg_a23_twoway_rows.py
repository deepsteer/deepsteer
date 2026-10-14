#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A16 (two-way bootstrap of record for sampled cells): the two dose-table rows
analyze_kdg_a23_twoway.py does not cover, for the v2 deliberation table.

    python3 papers/kdg_panel/scripts/analyze_kdg_a23_twoway_rows.py

On the same paired sets (OLMo-3 n 130, Llama-3.1 Meta n 114), primaries only: truncated filler minus
filler (P TF - P F) and brief reasoning minus filler (P D1 - P F, the 64-token arm), with the
scenario bootstrap and the two-way bootstrap (scenarios and rollout indices resampled independently,
one index set for every cell), 10,000 draws, seed 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as T  # noqa: E402
import analyze_kdg_a23_twoway as W  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    rep: dict = {"amendment": "P1-A16 (rows added for KDG v2)", "n_boot": A.N_BOOT, "seed": A.SEED,
                 "models": {}}
    for key, m in T.MODELS.items():
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ic, isp = C.llama_letter_ids(m["prim"])
        else:
            import analyze_continuous as ac

            ic, isp = ac._tok_ids()
        twin_ids = json.loads((A.DATA / "analysis_kdg_a23_twoway.json").read_text())
        cells = {"D1": (m["prim"], "d_chat_dose1_bf_forced"),
                 "F": (m["prim"], "d_chat_dose2_filler_bf_forced"),
                 "TF": (m["prim_tf"], "d_chat_dose2_filler_tf_forced"),
                 "D2": (m["prim"], "d_chat_dose2_bf_forced")}
        per = {k: T.per_scen(d, c, status, ic, isp) for k, (d, c) in cells.items()}
        screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        # the dose set of record: every forced cell defined (as the twin analysis selects it)
        twin = {c: T.per_scen(m["twin"], c, status, ic, isp)
                for c in (f"d_chat_dose2_bf{W.SUF}_forced", f"d_chat_dose2_filler_bf{W.SUF}_forced",
                          f"d_chat_dose2_filler_tf{W.SUF}_forced")}
        ids = sorted(s for s in screen if all(s in v for v in (*per.values(), *twin.values())))
        assert len(ids) == twin_ids["models"][key]["n"], (key, len(ids))
        X = {k: W.matrix(d, c, status, ic, isp, ids) for k, (d, c) in cells.items()}
        n, K = len(ids), W.K

        def stats(si, ri):
            mean = {k: X[k][np.ix_(si, ri)].mean(1) for k in X}
            return (mean["TF"] - mean["F"]).mean(), (mean["D1"] - mean["F"]).mean()

        point = stats(np.arange(n), np.arange(K))
        rng = np.random.default_rng(A.SEED)
        two = np.array([stats(rng.integers(0, n, n), rng.integers(0, K, K))
                        for _ in range(A.N_BOOT)])
        rng = np.random.default_rng(A.SEED)
        one = np.array([stats(rng.integers(0, n, n), np.arange(K)) for _ in range(A.N_BOOT)])
        rep["models"][key] = {"n": n, **{name: {"point": float(point[i]),
                                                "scenario_ci95": W.ci(one[:, i]),
                                                "two_way_ci95": W.ci(two[:, i])}
                                         for i, name in enumerate(("tf_minus_f", "d1_minus_f"))}}
        print(key, json.dumps(rep["models"][key]))
    (A.DATA / "analysis_kdg_a23_twoway_rows.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
