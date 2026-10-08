#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A11: dose-matched C0 (rule pushed 2d6057e before this script or any data).

    python3 papers/kdg_panel/scripts/analyze_gptoss_c0dm.py [--dm-dir <p2f cell dir>]

Reference: the forced primary argmax of the real run (``c0_forced_primary``, pod 3xlqmdx2mo3niz).
C0-dm: strict majority of the 4 generated answers after the empty closed analysis turn (ties and
re-deliberated or unparsed rollouts are non-matching; G-A1); pass iff agreement >= 0.80, Wilson CI
and near-miss label (G-A2); descriptive if more than a quarter of rollouts re-deliberate or
truncate.
Beside it (descriptive): C0-dm majority vs the low-effort majority (C0's reference), agreement and
the norm-crossing direction, a forced-readout-free second derivation of C0's 6-vs-0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

DM_DIR = A.OUT / "p2g" / "c0dm" / "gpt_oss_20b"  # p2g (combined); p2f: pass --dm-dir


def majorities(rows: list[dict], key=lambda r: r["reasoning_trace"] == "completed") -> dict:
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        if key(r):
            by[r["scenario_id"]].append(r["option_id"])  # None = unparsed or re-deliberated
    return by


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-dir", type=Path, default=DM_DIR)
    ap.add_argument("--ref-dir", type=Path, default=G.DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss_c0dm.json")
    a = ap.parse_args(argv)
    rows = A._rows(a.dm_dir / "c0dm_generate.jsonl")
    n = len(rows)
    re_rate = sum(r["redeliberated"] for r in rows) / n
    tr_rate = sum(r["reasoning_trace"] == "truncated" for r in rows) / n
    descriptive = re_rate + tr_rate > 0.25
    F = G.forced_argmax(a.ref_dir, "c0_forced_primary")
    dm = majorities(rows)
    ids = sorted(s for s in dm if s in F)
    verdict = kh.c0_verdict([kh.strict_majority(dm[s]) == F[s] for s in ids])
    if descriptive:
        branch = "descriptive"
    else:
        branch = "pass" if verdict["pass"] else "fail"

    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    label = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    low = majorities(A._rows(a.ref_dir / "c0_generate_low.jsonl"))
    both = [s for s in ids if s in low]
    agree, cross = 0, {"dm_violating_low_consistent": 0, "dm_consistent_low_violating": 0}
    for s in both:
        x, y = kh.strict_majority(dm[s]), kh.strict_majority(low[s])
        agree += x is not None and x == y
        if x is not None and y is not None and x != y:
            pair = (label[s].get(x), label[s].get(y))
            if pair == ("violating", "consistent"):
                cross["dm_violating_low_consistent"] += 1
            elif pair == ("consistent", "violating"):
                cross["dm_consistent_low_violating"] += 1
    rep = {
        "rule": "KDG_GPTOSS_SPEC G-A11 (2d6057e)", "n_rollouts": n,
        "redeliberated_rate": re_rate, "truncated_rate": tr_rate, "descriptive": descriptive,
        "parse_methods": dict(sorted({m: sum(r["parse_method"] == m for r in rows)
                                      for m in {r["parse_method"] for r in rows}}.items())),
        "c0_dm": verdict, "branch": branch,
        "descriptive_dm_vs_low_effort": {
            "n": len(both), "agreement": agree / len(both) if both else None,
            "wilson_ci95": list(kh.wilson(agree, len(both))), **cross,
        },
    }
    a.out.write_text(json.dumps(rep, indent=1))
    v = verdict
    print(f"C0-dm: {v['agreement']:.3f} Wilson {[round(x, 3) for x in v['wilson_ci95']]} -> "
          f"{v['verdict']}; re-deliberated {re_rate:.1%}, truncated {tr_rate:.1%} "
          f"-> branch {branch}")
    print("descriptive, dose-0 generation vs low-effort generation:",
          json.dumps(rep["descriptive_dm_vs_low_effort"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
