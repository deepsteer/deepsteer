#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A23 item 4 (descriptive): the KDG paper's deliberation brake by actor cost.

    python3 papers/kdg_panel/scripts/analyze_brake_actor_split.py [--labels <labels json>]

On each carrier's deliberation set of record (the scenarios ``analyze_dose_twin.py`` uses: OLMo-3
130, Llama-3.1 114), per scenario Delta_P = p_D(reasoning) - p_D(filler of record) on primaries.
Mean and bootstrap CI (10,000, seed 0) within actor-reversed, actor-aligned and split scenarios
under the actor-cost labels of record, and reversed minus aligned (independent resamples of the two
groups, seed 0). Actor-cost aversion predicts a weaker or reversed brake (Delta_P closer to 0 or
positive) in the actor-reversed scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as DT  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def diff_boot(a: np.ndarray, b: np.ndarray, seed: int = A.SEED) -> dict:
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(A.N_BOOT, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(A.N_BOOT, len(b)))].mean(axis=1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return {"mean": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)]}


def split(dP: dict[str, float], lab: dict) -> dict:
    rev, ali = set(lab["risk_reversed_ids"]), set(lab["risk_aligned_ids"])
    groups = {"reversed": [s for s in dP if s in rev], "aligned": [s for s in dP if s in ali],
              "split": [s for s in dP if s not in rev and s not in ali]}
    out = {g: A.boot(np.array([dP[s] for s in ids])) for g, ids in groups.items()}
    out["all"] = A.boot(np.array(list(dP.values())))
    if groups["reversed"] and groups["aligned"]:
        out["reversed_minus_aligned"] = diff_boot(np.array([dP[s] for s in groups["reversed"]]),
                                                  np.array([dP[s] for s in groups["aligned"]]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="actor_reversal_labels.json")
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_brake_actor_split.json")
    a = ap.parse_args(argv)
    lab = json.loads((A.DATA / a.labels).read_text())
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    rep: dict = {"rule": "KDG_GPTOSS_SPEC G-A23 item 4 (5ce8c22), descriptive",
                 "labels": a.labels, "n_boot": A.N_BOOT, "seed": A.SEED, "models": {}}
    for key, m in DT.MODELS.items():
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ids_chat, ids_sp = C.llama_letter_ids(m["prim"])
        else:
            import analyze_continuous as ac

            ids_chat, ids_sp = ac._tok_ids()
        D2 = DT.per_scen(m["prim"], "d_chat_dose2_bf_forced", status, ids_chat, ids_sp)
        F = DT.per_scen(m["prim"], "d_chat_dose2_filler_bf_forced", status, ids_chat, ids_sp)
        T = [DT.per_scen(m["twin"], f"d_chat_dose2{c}{DT.SUFFIX}_forced", status, ids_chat,
                         ids_sp) for c in ("_bf", "_filler_bf", "_filler_tf")]
        TF = DT.per_scen(m["prim_tf"], "d_chat_dose2_filler_tf_forced", status, ids_chat, ids_sp)
        screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
        # the dose set of record: every forced cell defined, as analyze_dose_twin selects it
        ids = sorted(s for s in screen if all(s in X for X in (D2, F, TF, *T)))
        dP = {s: D2[s] - F[s] for s in ids}
        rep["models"][key] = {"n": len(ids), **split(dP, lab)}
        r = rep["models"][key]
        print(f"{key:22s} n {len(ids)}: all {r['all']['mean']:+.3f}; reversed "
              f"{r['reversed']['mean']} (n {r['reversed']['n']}); aligned "
              f"{r['aligned']['mean']:+.3f} (n {r['aligned']['n']})")
    a.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
