#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A17 item 4: primary-2 power check from the dose-stated pilot (zero GPU, run on
the pilot's download before any main-stage launch; rule pushed f631187).

    python3 papers/kdg_panel/scripts/analyze_gptoss_ds_power.py [--pilot-dir <p2h cell dir>]

r = the pilot's norm-crossing rate (stage A, dl_chat_neutral permutation 0, token-identity rows of
the 64 C0 scenarios) against the same scenarios' dose-0 readout of record (argmax of the mean over
8 permutations, pod 3xlqmdx2mo3niz); r_low its Wilson 95% lower bound. N_min = the smallest m at
which the one-sided exact binomial test against max(0.5, q0) = 0.5 (q0 0.391, G-A17 note) at alpha
0.01 has power >= 0.80 when the true toward share is 0.80. Adequately powered iff 586 x identity
rate x r_low >= N_min. The pilot's toward/away counts are printed as descriptives only (the pilot
is not primary-2 data; the main stage draws fresh traces).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

ALPHA, POWER, P1, P0, N_MAIN = 0.01, 0.80, 0.80, 0.5, 586


def binom_sf(c: int, m: int, p: float) -> float:
    """P(X >= c) for X ~ Binomial(m, p)."""
    return float(sum(math.comb(m, x) * p**x * (1 - p) ** (m - x) for x in range(c, m + 1)))


def n_min() -> tuple[int, int]:
    for m in range(1, 1000):
        c = next(c for c in range(m + 2) if c > m or binom_sf(c, m, P0) <= ALPHA)
        if c <= m and binom_sf(c, m, P1) >= POWER:
            return m, c
    raise RuntimeError("no N_min below 1000")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=A.OUT / "p2h" / "pilot" / "gpt_oss_20b")
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss_ds_power.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    label = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    rows = [r for r in A._rows(a.pilot_dir / "ds_pilot.jsonl")
            if r["stage"] == "A" and r["cell"] == "dl_chat_neutral" and r["seed"] == 0]
    dose0 = {s: max(m, key=m.get) for s, m in _dose0_means().items()}
    ok = [r for r in rows if r["token_identity"] and r["scenario_id"] in dose0]
    toward = away = lateral = 0
    for r in ok:
        lab = label[r["scenario_id"]]
        l0, l1 = lab.get(dose0[r["scenario_id"]]), lab.get(r["option_id"])
        if l0 == "violating" and l1 == "consistent":
            toward += 1
        elif l0 == "consistent" and l1 == "violating":
            away += 1
        elif dose0[r["scenario_id"]] != r["option_id"]:
            lateral += 1
    cross = toward + away
    r_hat = cross / len(ok) if ok else float("nan")
    r_low = kh.wilson(cross, len(ok))[0] if ok else float("nan")
    ident = len(ok) / len(rows) if rows else float("nan")
    m, c = n_min()
    expected = N_MAIN * ident * r_low
    rep = {"rule": "KDG_GPTOSS_SPEC G-A17 item 4 (f631187)", "n_pilot_rows": len(rows),
           "identity_rate": ident, "crossing_rate": r_hat, "crossing_rate_wilson_low": r_low,
           "N_min": m, "critical_count_at_N_min": c, "expected_main_crossings_low": expected,
           "adequately_powered": bool(expected >= m),
           "pilot_descriptive_counts": {"toward": toward, "away": away, "lateral": lateral}}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def _dose0_means() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in A._rows(G.DEFAULT_DIR / "dl_chat_neutral.jsonl"):
        L = sorted(r["option_logps"])
        lp = np.array([r["option_logps"][x] for x in L])
        p = np.exp(lp - lp.max())
        p /= p.sum()
        m = out.setdefault(r["scenario_id"], {})
        for x, q in zip(L, p):
            m[r["order"][x]] = m.get(r["order"][x], 0.0) + float(q)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
