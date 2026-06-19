#!/usr/bin/env python3
"""Stage-1 control-vs-coupling comparison (attribution).

Both conditions run identical continued-pretrain; only the coupling term differs
(control = lambda 0). Subtracting the control isolates the regularizer's specific
effect from plain LoRA-on-general forgetting, so a held-out LM-quality change is
attributed correctly (the missing piece the step-0 guards alone can't supply).

Reads ``<dir>/control_<cap>/stage1_trajectory.json`` and ``<dir>/coupling_<cap>/
stage1_trajectory.json``; writes ``stage1_compare.json`` + ``STAGE1_COMPARE.md``.

Usage:
    python papers/5_moral_alignment/scripts/stage1_compare.py --capacity r16_qv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_DIR = _PAPER_ROOT / "outputs/intervention_stage1"


def _deltas(traj: dict) -> dict:
    r0, rL = traj["records"][0], traj["records"][-1]
    return {
        "proj_ref_0": r0["proj_refusal"], "proj_ref_final": rL["proj_refusal"],
        "d_proj": rL["proj_refusal"] - r0["proj_refusal"],
        "d_off": rL["proj_neutral_contrast"] - r0["proj_neutral_contrast"],
        "d_moral": rL["lm_moral"] - r0["lm_moral"],
        "d_neutral": rL["lm_neutral"] - r0["lm_neutral"],
        "d_general": rL["lm_general"] - r0["lm_general"],
    }


def _row(name: str, c: float, k: float, s: float, p: int = 3) -> str:
    return f"| {name} | {c:+.{p}f} | {k:+.{p}f} | {s:+.{p}f} |"


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-1 control-vs-coupling attribution.")
    ap.add_argument("--dir", default=str(_DEF_DIR))
    ap.add_argument("--capacity", default="r16_qv")
    ap.add_argument("--move-thresh", type=float, default=0.03,
                    help="Min coupling-specific projection move to count as 'moves'.")
    ap.add_argument("--harm-thresh", type=float, default=0.10,
                    help="Coupling-specific held-out LM rise (nats) that counts as harm.")
    args = ap.parse_args()

    d = Path(args.dir)
    cpath = d / f"control_{args.capacity}" / "stage1_trajectory.json"
    kpath = d / f"coupling_{args.capacity}" / "stage1_trajectory.json"
    if not cpath.exists() or not kpath.exists():
        print(f"Need both {cpath} and {kpath}; run with STAGE1_RUN_CONTROL=1.")
        return 1
    ctrl = _deltas(json.load(open(cpath)))
    coup = _deltas(json.load(open(kpath)))

    # Coupling-specific effects = coupling minus control.
    spec = {
        "proj_move": round(coup["d_proj"] - ctrl["d_proj"], 6),
        "offtarget_move": round(coup["d_off"] - ctrl["d_off"], 6),
        "moral_harm": round(coup["d_moral"] - ctrl["d_moral"], 6),
        "neutral_harm": round(coup["d_neutral"] - ctrl["d_neutral"], 6),
        "general_harm": round(coup["d_general"] - ctrl["d_general"], 6),
    }
    # §6 non-specificity = neutral harmed MORE than moral, beyond control.
    spec["neutral_minus_moral_harm"] = round(spec["neutral_harm"] - spec["moral_harm"], 6)

    moves = spec["proj_move"] >= args.move_thresh
    nonspecific = spec["neutral_minus_moral_harm"] >= args.harm_thresh
    broad_harm = max(spec["moral_harm"], spec["neutral_harm"]) >= args.harm_thresh
    offtarget_rise = spec["offtarget_move"] >= args.harm_thresh

    if not moves:
        verdict = "no_coupling_specific_move"
    elif nonspecific or offtarget_rise:
        verdict = "moves_but_degenerate"   # §6 recurs (moral-specific or off-target sink)
    elif broad_harm:
        verdict = "moves_with_broad_lm_cost"  # coupling-caused, but not moral-specific
    else:
        verdict = "moves_clean_vs_control"   # the green-light case

    payload = {"analysis": "stage1_compare", "capacity": args.capacity,
               "control": ctrl, "coupling": coup, "coupling_specific": spec,
               "verdict": verdict,
               "thresholds": {"move": args.move_thresh, "harm": args.harm_thresh}}
    with open(d / f"stage1_compare_{args.capacity}.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    md = [
        f"# Stage 1 control-vs-coupling ({args.capacity}) — {verdict}",
        "",
        "| metric | control Δ | coupling Δ | coupling-specific |",
        "|---|---|---|---|",
        _row("proj_refusal", ctrl["d_proj"], coup["d_proj"], spec["proj_move"], 4),
        _row("off-target", ctrl["d_off"], coup["d_off"], spec["offtarget_move"], 4),
        _row("lm_moral", ctrl["d_moral"], coup["d_moral"], spec["moral_harm"]),
        _row("lm_neutral", ctrl["d_neutral"], coup["d_neutral"], spec["neutral_harm"]),
        _row("lm_general", ctrl["d_general"], coup["d_general"], spec["general_harm"]),
        "",
        f"- proj_refusal {coup['proj_ref_0']:.4f} -> {coup['proj_ref_final']:.4f} "
        f"(coupling), control ends {ctrl['proj_ref_final']:.4f}",
        f"- §6 non-specificity (neutral harmed beyond moral, coupling-specific): "
        f"{spec['neutral_minus_moral_harm']:+.3f}",
        "",
        "Routing: no_coupling_specific_move -> capacity too weak (climb); "
        "moves_clean_vs_control -> green-light Stage 2; moves_with_broad_lm_cost -> "
        "coupling degrades LM broadly (not moral-specific); moves_but_degenerate -> "
        "§6 recurs (neutral-specific harm or off-target sink).",
    ]
    (d / f"STAGE1_COMPARE_{args.capacity}.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {d}/stage1_compare_{args.capacity}.json and STAGE1_COMPARE_{args.capacity}.md")
    print(f"  VERDICT: {verdict} | coupling-specific proj move {spec['proj_move']:+.4f}, "
          f"neutral-minus-moral harm {spec['neutral_minus_moral_harm']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
