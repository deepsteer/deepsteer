#!/usr/bin/env python3
"""KDG-A12 discriminator (KDG_PHASE1_SPEC.md P1-A10, pushed 04d604f before this ran): the stage
steps of the pressure-attributable excess per unit of output scale, on the P1-A7 final-model-free
set.

    python3 papers/kdg_panel/scripts/analyze_kdg_a12.py

Primary: the RL step of E_norm (E_logit / averaged twin sigma, KDG-44's scale). E_fs and E_logit
beside; E_prob re-reported. Tulu 3 beside, descriptive. Every bound unrounded, 10,000 resamples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_kdg_a8 as K  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_phase1_session_b as B  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

QS = ("E_prob", "E_norm", "E_fs", "E_logit")
AT_BAR_PROB = 0.001  # P1-A10: a lower bound within 0.001 of 0 on E_prob is at-the-bar
BAR_PROB_RL = 0.005  # the RL-step E_prob bar of record (KDG-44)


def se(b: dict) -> float:
    lo, hi = b["ci95"]
    return (hi - lo) / (2 * 1.96)


def profile(T: dict, ids: list[str], stages: tuple[str, ...]) -> dict:
    S = {st: [A.scales(T[st][s]) for s in ids] for st in stages}
    rep = {"n": len(ids), "per_stage": {}, "steps": {}}
    for q in QS:
        rep["per_stage"][q] = {st: A.boot(np.array([x[q] for x in S[st]])) for st in stages}
        rep["steps"][q] = {
            n: A.boot(np.array([y[q] - x[q] for x, y in zip(S[a], S[b])]))
            for a, b, n in (("sft", "dpo", "DPO"), ("dpo", "final", "RL"))
        }
    return rep, S


def verdict(steps: dict, q: str) -> str:
    """P1-A10 rule on the RL step of per-scale quantity q, against E_prob's standardized size."""
    b, p = steps[q]["RL"], steps["E_prob"]["RL"]
    if A.sign_of(b["ci95"]) > 0:
        return "survives_scale"
    z_q, z_p = b["mean"] / se(b), p["mean"] / se(p)
    return "sharpening_explained" if z_q < 0.5 * z_p else "unresolved"


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    C = {st: A.four_cells(d, K.NAMES, status) for st, d in K.DIRS.items()}
    ids = sorted(s for s in status
                 if all(s in C[st] and C[st][s]["mass_min"] >= A.FLOOR for st in C))
    a8 = json.loads((A.DATA / "analysis_kdg_a8.json").read_text())
    # most probable failure: the set drifts from P1-A7's (different floor or exclusions), so the
    # per-scale step is read on a different population than the probability step it tests
    assert len(ids) == a8["n"] == 586, len(ids)
    olmo, S = profile(C, ids, ("sft", "dpo", "final"))
    assert abs(olmo["steps"]["E_prob"]["RL"]["mean"] - a8["steps"]["E"]["RL"]["mean"]) < 1e-12

    v_norm, v_fs = verdict(olmo["steps"], "E_norm"), verdict(olmo["steps"], "E_fs")
    order = ("survives_scale", "unresolved", "sharpening_explained")
    primary = v_norm if v_norm == v_fs else max(v_norm, v_fs, key=order.index)
    rl = olmo["steps"]["E_norm"]["RL"]
    mde_norm = rl["mde"]
    at_bar = {
        "E_prob": abs(olmo["steps"]["E_prob"]["RL"]["ci95"][0]) < AT_BAR_PROB,
        "E_norm": abs(rl["ci95"][0]) < (AT_BAR_PROB / BAR_PROB_RL) * mde_norm,
    }
    stab = {}
    for q in ("E_prob", "E_norm"):
        d = np.array([y[q] - x[q] for x, y in zip(S["dpo"], S["final"])])
        stab[q] = [A.boot(d, seed=k)["ci95"][0] for k in range(10)]

    TC = {st: B.cells(k, "chat", status) for st, k in
          (("sft", "tulu3_sft"), ("dpo", "tulu3_dpo"), ("final", "tulu3_final"))}
    t_ids = sorted(s for s in status if all(B.ok(TC[st], s) for st in TC))
    tulu, _ = profile(TC, t_ids, ("sft", "dpo", "final"))

    rep = {
        "amendment": "P1-A10 (04d604f)", "n_boot": A.N_BOOT, "seed": A.SEED,
        "olmo3": olmo,
        "verdict_rl": {"E_norm": v_norm, "E_fs": v_fs, "reading": primary,
                       "scale_dependent": v_norm != v_fs},
        "z_rl": {q: olmo["steps"][q]["RL"]["mean"] / se(olmo["steps"][q]["RL"]) for q in QS},
        "at_the_bar_rl": at_bar,
        "stability_rl_lower_bound_seeds_0_9": stab,
        "tulu3_descriptive": tulu,
    }
    (A.DATA / "analysis_kdg_a12.json").write_text(json.dumps(rep, indent=1))

    f = lambda x: "%+.5f [%+.5f, %+.5f] mde %.4f" % (x["mean"], *x["ci95"], x["mde"])  # noqa: E731
    print("OLMo-3 n", olmo["n"], "| Tulu 3 n", tulu["n"])
    for q in QS:
        print(f"{q:8s} DPO {f(olmo['steps'][q]['DPO'])}   RL {f(olmo['steps'][q]['RL'])}")
    print("z (RL)", {k: round(v, 2) for k, v in rep["z_rl"].items()})
    print("verdict RL", rep["verdict_rl"], "at-the-bar", at_bar)
    print("stability lower bounds", {k: [round(x, 5) for x in v] for k, v in stab.items()})
    print("per stage E_norm", {k: f(v) for k, v in olmo["per_stage"]["E_norm"].items()})
    for q in QS:
        print(f"Tulu {q:8s} DPO {f(tulu['steps'][q]['DPO'])}   RL {f(tulu['steps'][q]['RL'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
