#!/usr/bin/env python3
"""Zero-GPU discriminators for ANOMALIES KDG-A1..A3 from the saved pilot arrays.

    python3 papers/kdg_panel/scripts/analyze_anomalies.py --out papers/kdg_panel/outputs/pilot

KDG-A1: consideration breadth (length-residualized) on J_stated greedy replies, stable vs
        unstable scenarios (needs ``j_stated_breadth.json`` from ``rate_with_judge.py breadth``).
KDG-A2: top-5 next tokens on the below-floor instruct raw rows (formatting vs refusal tokens),
        decoded with the OLMo-3 tokenizer; enrichment of the below-floor set for D_chat pressure.
KDG-A3: rollout-level violating fraction per family vs the majority-rule KDG per family.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_pilot import BASE_FLOOR, _by_scenario, _rows, build_readouts  # noqa: E402

from deepsteer.kdg.stats import length_residualize, screen_pass  # noqa: E402


def _tokenizer():
    from transformers import AutoTokenizer

    snaps = sorted(
        (Path.home() / ".cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct/snapshots").glob(
            "*/tokenizer.json"
        )
    )
    return AutoTokenizer.from_pretrained(str(snaps[-1].parent))


def a1_breadth(inst: Path, ro) -> dict:
    p = inst / "j_stated_breadth.json"
    if not p.exists():
        return {"status": "pending: run rate_with_judge.py breadth on j_stated.jsonl"}
    items = json.loads(p.read_text())["items"]
    scored = [it for it in items if it.get("breadth") is not None]
    b = np.array([it["breadth"] for it in scored], float)
    n = np.array([it["n_words"] for it in scored], float)
    resid = length_residualize(b, n)
    res_by = defaultdict(list)
    for it, r in zip(scored, resid):
        res_by[it["scenario_id"]].append(r)
    stab = {r.scenario_id: r.judgment_stable() for r in ro}
    st = [np.mean(res_by[s]) for s in res_by if stab.get(s)]
    un = [np.mean(res_by[s]) for s in res_by if s in stab and not stab[s]]
    raw_st = [
        np.mean(
            [
                it["breadth"]
                for it in items
                if it["scenario_id"] == s and it.get("breadth") is not None
            ]
        )
        for s in res_by
        if stab.get(s)
    ]
    raw_un = [
        np.mean(
            [
                it["breadth"]
                for it in items
                if it["scenario_id"] == s and it.get("breadth") is not None
            ]
        )
        for s in res_by
        if s in stab and not stab[s]
    ]
    rng = np.random.default_rng(0)
    diffs = (
        [np.mean(rng.choice(st, len(st))) - np.mean(rng.choice(un, len(un))) for _ in range(2000)]
        if st and un
        else []
    )
    return {
        "n_scored": len(scored),
        "n_stable_scenarios": len(st),
        "n_unstable_scenarios": len(un),
        "mean_breadth_raw_stable": float(np.mean(raw_st)) if raw_st else None,
        "mean_breadth_raw_unstable": float(np.mean(raw_un)) if raw_un else None,
        "mean_residual_stable": float(np.mean(st)) if st else None,
        "mean_residual_unstable": float(np.mean(un)) if un else None,
        "residual_diff_stable_minus_unstable": float(np.mean(diffs)) if diffs else None,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        if diffs
        else None,
        "reading": "R_a predicts unstable scenarios carry HIGHER breadth (many considerations); "
        "R_b predicts no difference",
    }


def a2_tokens(out: Path, floor: float = BASE_FLOOR) -> dict:
    tok = _tokenizer()
    res = {}
    dchat = _by_scenario(_rows(out / "olmo3_instruct" / "d_chat_dose0.jsonl"))
    for key in ("olmo3_instruct", "olmo3_base"):
        rows = _rows(out / key / "d_raw.jsonl")
        z = np.load(out / key / "d_raw.npz")
        lp = z["logp_decision"].astype(np.float32)
        below = [i for i, r in enumerate(rows) if r["option_mass"] < floor]
        above = [i for i, r in enumerate(rows) if r["option_mass"] >= floor]
        top = defaultdict(float)
        for i in below:
            for t in np.argsort(-lp[i])[:5]:
                top[tok.decode([int(t)])] += float(np.exp(lp[i][t]))
        top5 = sorted(top.items(), key=lambda kv: -kv[1])[:12]
        # enrichment: D_chat violating fraction of scenarios with any below-floor row vs none
        bf_ids = {rows[i]["scenario_id"] for i in below}

        def vf(sid):
            st = [r["norm_status"] for r in dchat.get(sid, [])]
            return np.mean([s == "violating" for s in st if s]) if st else np.nan

        v_bf = [vf(s) for s in bf_ids if s in dchat]
        v_ok = [vf(s) for s in {r["scenario_id"] for r in rows} - bf_ids if s in dchat]
        res[key] = {
            "n_rows": len(rows),
            "n_below_floor_rows": len(below),
            "n_scenarios_below": len(bf_ids),
            "mean_mass_below": float(np.mean([rows[i]["option_mass"] for i in below]))
            if below
            else None,
            "mean_mass_above": float(np.mean([rows[i]["option_mass"] for i in above]))
            if above
            else None,
            "top_tokens_below_floor(prob mass summed)": [(t, round(m, 2)) for t, m in top5],
            "d_chat_violating_frac_scenarios_with_below_floor": float(np.nanmean(v_bf))
            if v_bf
            else None,
            "d_chat_violating_frac_other_scenarios": float(np.nanmean(v_ok)) if v_ok else None,
        }
    return res


def a3_rollout_family(ro) -> dict:
    scr = {r.scenario_id: screen_pass(r)[0] for r in ro}
    out = {}
    for f in ("F1", "F2", "F3", "F4", "F5"):
        rs = [r for r in ro if r.family == f and scr[r.scenario_id] and r.kdg() is not None]
        nv = [r for r in rs if r.j_greedy != "violating"]
        out[f] = {
            "n_screened_defined": len(rs),
            "majority_kdg": float(np.mean([r.kdg() for r in rs])) if rs else None,
            "rollout_violating_frac_given_nonviolating_J": float(
                np.mean([r.d_violating_fraction() for r in nv])
            )
            if nv
            else None,
            "n_nonviolating_J": len(nv),
        }
    return {
        "per_family": out,
        "reading": "R_b (noise) predicts the F3 majority excess vanishes at the rollout level",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "papers/kdg_panel/outputs/pilot")
    a = ap.parse_args()
    inst = a.out / "olmo3_instruct"
    ro = build_readouts(inst)
    rep = {
        "KDG-A1": a1_breadth(inst, ro),
        "KDG-A2": a2_tokens(a.out),
        "KDG-A3": a3_rollout_family(ro),
    }
    (a.out / "analysis_anomalies.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
