#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU): multi-source GATE G2 coverage (fables + ETHICS).

Extends the single-source G2 (phase2_g2.py: Moral Stories narrative PASS) to the two added
V_moral sources. For each source, projects that source's HELD-OUT eval pairs onto that
source's BASE mean-diff direction and computes paired transfer accuracy on surface vs clean
paraphrase (acc_surf, acc_para via direction_utils.transfer_metrics.acc_midpoint). The gap
acc_surf - acc_para is the contamination signal: a large drop means the direction reads
memorized surface text, not moral structure.

Per-source held-out separation (verified disjoint):
  * fables  -- d_fables from the 62 TRAIN-split pairs; eval = 15 held-out EVAL-split pairs.
  * ethics  -- d_ethics from the 118 probe pairs; held-out eval = 199 disjoint TRAIN-split
               pairs (overlap 0 by id). The 118 EXTRACTION pairs are ALSO reported as a
               separate extraction-pair paraphrase-gap diagnostic (NOT a held-out G2; this is
               the user's named follow-up, kept distinct).

The G2 gate threshold (acc_para>=0.60, gap<=0.10) is narrative-calibrated, so it GATES the
narrative slice (fables) and is reported INFORMATIVE for declarative slices (ETHICS). Per the
settled G2<->G3 distinction (RESULTS.md): a soft per-source G2 is a statement about that
source's training-pair contamination, NOT a threat to the G3 orthogonality headline (which
rests on the source mean-diff directions + the rank-matched null). This script never raises on
a soft number; it reports. GPU; VALIDATE=1 = tiny plumbing smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
import direction_utils as du  # noqa: E402

FLOOR = 0.60
MAX_GAP = 0.10
MATCH_LAYER = 16
_FULL = HERE.parent / "outputs" / "full"
_AXIS = HERE.parent / "outputs" / "phase2" / "axis" / "axis_directions.npz"

# name -> (direction key in axis_directions.npz, paraphrased eval file, register, kind, gated)
SOURCES = [
    ("fables", "fables", _FULL / "fables_eval_paraphrased.json",
     "narrative", "held-out", True),
    ("ethics_heldout", "ethics", _FULL / "ethics_heldout_paraphrased.json",
     "declarative", "held-out", False),
    ("ethics_extraction", "ethics", _FULL / "ethics_eval_paraphrased.json",
     "declarative", "extraction-gap", False),
]


def clean_rows(path: Path) -> list[dict]:
    d = json.load(open(path))
    rows = d["pairs"] if isinstance(d, dict) and "pairs" in d else d
    return [r for r in rows if r.get("status") == "clean"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-source GATE G2 (fables + ETHICS).")
    ap.add_argument("--axis", default=str(_AXIS))
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "phase2"
                                         / "g2_multisource_result.json"))
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model = "allenai/OLMo-2-0425-1B"

    axis_dirs = du.load_directions(Path(args.axis)) if Path(args.axis).exists() else {}

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    L = min(MATCH_LAYER, model.info.n_layers - 1)

    report: list[dict] = []
    for name, dir_key, para_file, register, kind, gated in SOURCES:
        if not para_file.exists():
            report.append({"source": name, "skipped": f"missing {para_file.name}"})
            continue
        rows = clean_rows(para_file)
        if validate:
            rows = rows[:8]
        surf = [(r["moral"], r["neutral"]) for r in rows]
        para = [(r["moral_para"], r["neutral_para"]) for r in rows]
        Xs, y = du.collect_pair_activations(model, surf, input_format="raw", layers=[L])[L]
        Xp, _ = du.collect_pair_activations(model, para, input_format="raw", layers=[L])[L]

        # Real run: the frozen base source direction. Smoke (dim-mismatched 1B): derive from
        # surf so the plumbing exercises end-to-end without a 4096-vs-2048 mismatch.
        d = axis_dirs.get(dir_key, {}).get(L)
        if d is None or d.shape[0] != (Xs.shape[1] if hasattr(Xs, "shape") else len(Xs[0])):
            d = du.mean_diff_direction(Xs, y)

        acc_surf = du.transfer_metrics(Xs, y, d)["acc_midpoint"]
        acc_para = du.transfer_metrics(Xp, y, d)["acc_midpoint"]
        gap = acc_surf - acc_para
        passed = bool(acc_para >= FLOOR and gap <= MAX_GAP)
        report.append({
            "source": name, "direction": dir_key, "register": register, "kind": kind,
            "gated": gated, "n_pairs": len(rows),
            "acc_surf": round(float(acc_surf), 4), "acc_para": round(float(acc_para), 4),
            "gap": round(float(gap), 4),
            "verdict": ("PASS" if passed else "SOFT") if gated else "INFORMATIVE",
        })
    model.release()

    result = {"gate": "G2-multisource", "floor": FLOOR, "max_gap": MAX_GAP, "layer": L,
              "model": args.model, "sources": report,
              "note": "Per the G2<->G3 distinction (RESULTS.md), a SOFT per-source G2 is a "
                      "training-pair contamination statement, not a threat to the G3 headline. "
                      "Gate threshold is narrative-calibrated; declarative slices are "
                      "informative. ethics_extraction is the extraction-pair paraphrase-gap "
                      "diagnostic (NOT a held-out G2)."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    for r in report:
        if "skipped" in r:
            print(f"  {r['source']:<18} SKIPPED ({r['skipped']})")
        else:
            print(f"  {r['source']:<18} [{r['kind']:<13} {r['register']:<11}] "
                  f"n={r['n_pairs']:<3} acc_surf={r['acc_surf']:.3f} "
                  f"acc_para={r['acc_para']:.3f} gap={r['gap']:+.3f} -> {r['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
