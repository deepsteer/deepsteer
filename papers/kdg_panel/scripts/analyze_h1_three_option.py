#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 5: the carriers' three-option movement on H1 (descriptive).

    python3 papers/kdg_panel/scripts/analyze_h1_three_option.py

Per carrier (pod p1e), the forced readout's mass on the rule-following option (norm-consistent),
the neutral option (log and contact the approver) and the protective departure (violating),
averaged over rollouts per scenario and then over the 264 scenarios, for the reasoning arm and the
filler of record; the paired change per option class with a scenario bootstrap (10,000, seed 0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h1_carriers as HC  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

CLASSES = ("consistent", "neutral", "violating")


def option_masses(d: Path, cell: str, status: dict, ic, isp) -> dict[str, np.ndarray]:
    rows = A._rows(d / f"{cell}.jsonl")
    lp = np.load(d / f"{cell}.npz")["logp_decision"]
    acc: dict[str, list] = {}
    for k, r in enumerate(rows):
        sid = r["scenario_id"]
        if sid not in status:
            continue
        L = sorted(r["order"])
        vec = lp[k].astype(np.float64)
        m = np.exp(vec[[ic[x] for x in L]]) + np.exp(vec[[isp[x] for x in L]])
        m = m / m.sum()
        v = np.zeros(3)
        for x, q in zip(L, m):
            v[CLASSES.index(status[sid][r["order"][x]])] += q
        acc.setdefault(sid, []).append(v)
    return {s: np.mean(v, axis=0) for s, v in acc.items()}


def main() -> int:
    H, _ = load_scenario_dir([A.DATA / "harm_divergence" / "hdiv_set.json"])
    ids = set(json.loads((A.DATA / "harm_divergence" / "h1_kept_ids.json").read_text())["ids"])
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in H if s.id in ids}
    rep: dict = {"rule": "KDG_PHASE1_SPEC P1-A17 item 5, descriptive", "models": {}}
    for key, step in HC.MODELS.items():
        d = A.OUT / "p1e" / step / key
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ic, isp = C.llama_letter_ids(d)
        else:
            import analyze_continuous as ac

            ic, isp = ac._tok_ids()
        R = option_masses(d, "d_chat_dose2_bf_forced", status, ic, isp)
        F = option_masses(d, "d_chat_dose2_filler_bf_forced", status, ic, isp)
        both = sorted(s for s in R if s in F)
        r, f = np.array([R[s] for s in both]), np.array([F[s] for s in both])
        rep["models"][key] = {"n": len(both), **{c: {
            "filler": float(f[:, i].mean()), "reasoning": float(r[:, i].mean()),
            "change": A.boot(r[:, i] - f[:, i])} for i, c in enumerate(CLASSES)}}
        print(key, {c: (round(v["filler"], 3), round(v["reasoning"], 3),
                        round(v["change"]["mean"], 3), [round(x, 3) for x in v["change"]["ci95"]])
                    for c, v in rep["models"][key].items() if c != "n"})
    (A.DATA / "analysis_h1_three_option.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
