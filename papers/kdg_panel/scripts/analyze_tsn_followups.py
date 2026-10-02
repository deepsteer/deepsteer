#!/usr/bin/env python3
"""Turns-since-norm follow-ups (KDG_F6_F8_SPEC.md P2-A4, pushed 00eb2b2 before any cell ran).

    python3 papers/kdg_panel/scripts/analyze_tsn_followups.py --out papers/kdg_panel/outputs/p2b/...

A4a (Llama, counterbalanced filler order): rotated R(3) and R(6) vs the fixed-order R(3) of record
(0.684 [0.603, 0.771]). Filler confound (R_a) iff the rotated R(3) CI includes the rotated R(6)
point and lies above 0.771; position effect (R_b) iff the rotated R(3) CI lies entirely below the
rotated R(6) CI; unresolved otherwise.

A4b (both models, token-distance ladder at matched turn count): R_L(6) = mean Δ_L(6) / mean Δ_L(0)
for L in 600 and 2,000 (138 from p2a). Precondition per L: Δ_L(0) CI below 0. Decays with token
distance iff R_2000(6) < 1 with its CI excluding 1; else no decay detectable to 2,000 tokens, with
the bar MDE(1 − R_2000(6)) = 2.8 · SD(Δ(6) − Δ(0)) / (√n · |mean Δ(0)|).
Same p_D, floor and bootstrap (10,000, seed 0) as analyze_tsn.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_tsn as T  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

FIXED_R3_UPPER = 0.771  # KDG-A16, fixed-order R(3) upper bound on Llama (analysis_tsn.json)


def cell(kind: str, k: int, pr: bool, length: int = 138, rotate: bool = False) -> str:
    tag = "dl_tsn" + ("" if length == 138 else str(length)) + ("rot" if rotate else "")
    return f"{tag}_{kind}_k{k}" + ("_pressure_removed" if pr else "")


def deltas(d: Path, ids: set[str], status: dict, ks, length: int, rotate: bool):
    C = {
        (kd, k, pr): T.read(d, cell(kd, k, pr, length, rotate), status)
        for kd in ("reminder", "neutral")
        for k in ks
        for pr in (False, True)
    }
    use = sorted(s for s in ids if all(s in X and X[s][1] >= T.FLOOR for X in C.values()))
    dl = {
        k: np.array([C[("reminder", k, False)][s][0] - C[("neutral", k, False)][s][0] for s in use])
        for k in ks
    }
    return use, dl


def a4a(d: Path, ids, status) -> dict:
    use, dl = deltas(d, ids, status, (0, 3, 6), 138, True)
    r3, r6 = T.boot_ratio(dl[3], dl[0]), T.boot_ratio(dl[6], dl[0])
    if r3["ci95"][0] <= r6["ratio"] <= r3["ci95"][1] and r3["ci95"][0] > FIXED_R3_UPPER:
        v = "filler_confound_R_a"
    elif r3["ci95"][1] < r6["ci95"][0]:
        v = "position_effect_R_b"
    else:
        v = "unresolved"
    return {
        "n": len(use),
        "delta": {k: T.boot(x) for k, x in dl.items()},
        "R3": r3,
        "R6": r6,
        "verdict": v,
    }


def a4b(d: Path, ids, status, length: int) -> dict:
    use, dl = deltas(d, ids, status, (0, 6), length, False)
    rep: dict = {"n": len(use), "delta": {k: T.boot(x) for k, x in dl.items()}}
    if not rep["delta"][0]["ci95"][1] < 0:
        rep["verdict"] = "no_reminder_effect"
        return rep
    rep["R6"] = T.boot_ratio(dl[6], dl[0])
    sd = float((dl[6] - dl[0]).std(ddof=1))
    rep["mde_one_minus_R6"] = 2.8 * sd / (np.sqrt(len(use)) * abs(rep["delta"][0]["mean"]))
    r = rep["R6"]
    rep["verdict"] = "decays" if (r["ratio"] < 1 and r["ci95"][1] < 1) else "no_decay_detectable"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(T.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    out: dict = {"amendment": "P2-A4 (00eb2b2)", "n_boot": T.N_BOOT, "seed": T.SEED, "models": {}}
    for m, screen in T.SCREENS.items():
        d = T.model_dir(a.out, m)
        if d is None:
            continue
        raw = json.loads(screen.read_text())
        ids = set(raw["ids"] if isinstance(raw, dict) else raw) & set(status)
        r: dict = {}
        if (d / f"{cell('reminder', 3, False, 138, True)}.jsonl").exists():
            r["A4a"] = a4a(d, ids, status)
        for L in (600, 2000):
            if (d / f"{cell('reminder', 0, False, L)}.jsonl").exists():
                r[f"A4b_{L}"] = a4b(d, ids, status, L)
        out["models"][m] = r
    path = a.report or (T.DATA / "analysis_tsn_followups.json")
    path.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {m: {k: v.get("verdict") for k, v in r.items()} for m, r in out["models"].items()},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
