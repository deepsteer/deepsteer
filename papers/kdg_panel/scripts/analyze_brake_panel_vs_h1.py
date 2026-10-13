#!/usr/bin/env python3
"""Brake magnitude, panel vs H1, per carrier (author request 2026-10-13; descriptive comparison).

    python3 papers/kdg_panel/scripts/analyze_brake_panel_vs_h1.py

Per scenario Delta_P = p_D(reasoning, 512-token forced arm) - p_D(filler of record). Panel: the
dose set of record (as analyze_dose_twin; the violating option is the harmful one, so the rule
is the protective choice). H1 (pod p1e): the 264 kept scenarios (the violating option is the
protective departure, so the rule is the harmful choice). Reported: both means, the difference
(panel - H1) with a bootstrap CI over independent resamples of the two scenario sets (10,000,
seed 0), and the detectable difference 2.8 x SE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as DT  # noqa: E402
import analyze_h1_carriers as HC  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def deltas(d: Path, status: dict, ids: set, key: str) -> np.ndarray:
    if key.startswith("llama"):
        import analyze_phase1_session_c as C

        ic, isp = C.llama_letter_ids(d)
    else:
        import analyze_continuous as ac

        ic, isp = ac._tok_ids()
    D2 = DT.per_scen(d, "d_chat_dose2_bf_forced", status, ic, isp)
    F = DT.per_scen(d, "d_chat_dose2_filler_bf_forced", status, ic, isp)
    return np.array([D2[s] - F[s] for s in sorted(ids) if s in D2 and s in F])


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    pstat = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
             if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    H, _ = load_scenario_dir([A.DATA / "harm_divergence" / "hdiv_set.json"])
    hids = set(json.loads((A.DATA / "harm_divergence" / "h1_kept_ids.json").read_text())["ids"])
    hstat = {s.id: {o.option_id: o.norm_status for o in s.options} for s in H if s.id in hids}
    rep: dict = {"rule": "author request 2026-10-13, descriptive", "n_boot": A.N_BOOT,
                 "seed": A.SEED, "models": {}}
    rng = np.random.default_rng(A.SEED)
    for key, m in DT.MODELS.items():
        hd = A.OUT / "p1e" / HC.MODELS[key] / key
        if not (hd / "d_chat_dose2_bf_forced.jsonl").exists():
            rep["models"][key] = {"status": "H1 not run yet"}
            continue
        pids = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        p = deltas(m["prim"], pstat, pids, key)
        h = deltas(hd, hstat, hids, key)
        bp = p[rng.integers(0, len(p), size=(A.N_BOOT, len(p)))].mean(axis=1)
        bh = h[rng.integers(0, len(h), size=(A.N_BOOT, len(h)))].mean(axis=1)
        diff = bp - bh
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rep["models"][key] = {"panel": {"n": len(p), "mean": float(p.mean())},
                              "h1": {"n": len(h), "mean": float(h.mean())},
                              "difference_panel_minus_h1": float(p.mean() - h.mean()),
                              "ci95": [float(lo), float(hi)],
                              "detectable_2p8se": float(2.8 * diff.std(ddof=1))}
        print(key, json.dumps(rep["models"][key]))
    (A.DATA / "analysis_brake_panel_vs_h1.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
