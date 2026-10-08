#!/usr/bin/env python3
"""ANOMALIES KDG-A20 item 2: Qwen2.5 C3 re-read vs the cells of record (rule pushed 3db23c8).

    python3 papers/kdg_panel/scripts/analyze_kdg_a20_qwen_reread.py

Record: the four C3 letter cells of Qwen2.5-7B-Instruct (readout version 1). Re-read: the same four
cells, same scenario order and batch size, readout version 2, on the stack of record (profile p2e).
Quantity: per-scenario paired ΔE_s = E_s(re-read) − E_s(record) on the record's model-free ids (not
re-screened); paired bootstrap 10,000, seed 0; per-row |Δ log p| on the option tokens (max, p99).

E of record: the pre-registration points to ``analysis_continuous_union.json``, which carries no
Qwen2.5 entry; the value is recomputed here from the record cells with the same definition
(−0.0076 [−0.0235, 0.0088], n 586; KDG-A19 prints −0.008 [−0.023, 0.009]).

Branches, applied in their pre-registered order (fixed before data, 2026-10-07): **stands** if the
ΔE CI lies inside ±0.005; else **correction** if it excludes 0; else **unresolved**. A CI that
excludes 0 but lies inside ±0.005 therefore reads "stands" (the listed order; the overlap is noted).
Beside it (descriptive, KDG-A19): own-screen E and its selection-matched null under readout v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

REREAD = A.OUT / "p2e" / "reread" / "qwen25_instruct_p1"
MARGIN = 0.005


def branch(ci: list[float], margin: float = MARGIN) -> str:
    lo, hi = ci
    if -margin <= lo and hi <= margin:
        return "stands"
    if lo > 0 or hi < 0:
        return "correction"
    return "unresolved"


def own_screen(d: Path, status: dict) -> dict:
    T = A.four_cells([d], OS.CHAT, status)
    own = [s for s in OS.screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
    tw = [s for s in OS.twin_screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
    e_own = np.array([A.scales(T[s])["E_prob"] for s in own])
    e_rev = np.array([-A.scales(T[s])["E_prob"] for s in tw])
    rng = np.random.default_rng(A.SEED)
    diff = (e_own[rng.integers(0, len(e_own), (A.N_BOOT, len(e_own)))].mean(1)
            - e_rev[rng.integers(0, len(e_rev), (A.N_BOOT, len(e_rev)))].mean(1))
    return {"n_own": len(own), "E_own": A.boot(e_own), "n_twin_screen": len(tw),
            "E_sel": float(e_own.mean() - e_rev.mean()),
            "E_sel_ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, default=SR.MODELS["qwen25_instruct"])
    ap.add_argument("--reread", type=Path, default=REREAD)
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_kdg_a20_qwen_reread.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    for d, want in ((a.record, {1}), (a.reread, {2})):
        got = set().union(*(A.readout_versions([d], c) for c in OS.CHAT))
        assert got == want, f"{d}: readout versions {got}, expected {want}"
    R = A.four_cells([a.record], OS.CHAT, status)
    N = A.four_cells([a.reread], OS.CHAT, status)
    ids = sorted(s for s in R if R[s]["mass_min"] >= A.FLOOR)
    missing = [s for s in ids if s not in N]
    assert not missing, f"{len(missing)} record ids absent from the re-read"
    e_rec = np.array([A.scales(R[s])["E_prob"] for s in ids])
    e_new = np.array([A.scales(N[s])["E_prob"] for s in ids])
    dE = A.boot(e_new - e_rec)
    dlp = []
    for c in OS.CHAT:
        old = {(r["scenario_id"], r["seed"]): r["option_logps"]
               for r in A._rows(a.record / f"{c}.jsonl")}
        for r in A._rows(a.reread / f"{c}.jsonl"):
            o = old[(r["scenario_id"], r["seed"])]
            dlp.append(max(abs(o[L] - r["option_logps"][L]) for L in o))
    rep = {
        "rule": "ANOMALIES KDG-A20 item 2 (3db23c8); branch order fixed before data",
        "n": len(ids), "E_record": A.boot(e_rec), "E_reread": A.boot(e_new), "dE": dE,
        "row_abs_dlogp": {"max": float(np.max(dlp)), "p99": float(np.percentile(dlp, 99)),
                          "median": float(np.median(dlp)), "n_rows": len(dlp)},
        "branch": branch(dE["ci95"]),
        "descriptive_kdg_a19": {"record": own_screen(a.record, status),
                                "reread": own_screen(a.reread, status)},
    }
    a.out.write_text(json.dumps(rep, indent=1))
    f = lambda x: "%+.4f [%+.4f, %+.4f]" % (x["mean"], *x["ci95"])  # noqa: E731
    print(f"E record {f(rep['E_record'])}  re-read {f(rep['E_reread'])}  dE {f(dE)} "
          f"-> {rep['branch']}; row |dlogp| max {rep['row_abs_dlogp']['max']:.3f} "
          f"p99 {rep['row_abs_dlogp']['p99']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
