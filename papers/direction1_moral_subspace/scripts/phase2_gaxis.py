#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 1: GATE G-AXIS (two-source agreement).

Reads the per-source mean-diff directions from stage 0 and decides MORABLES pooling by the
pre-registered floor 0.67 (PREREGISTRATION §3A). Writes `g_axis_decision.json`, the artifact
that stage 2 branches on. BOTH branches are real:

  * cos >= 0.67  -> PASS: V_moral sources = [moral_stories, morables]   (two-source)
  * cos <  0.67  -> FAIL: V_moral sources = [moral_stories]             (single-source fallback)

A FAIL triggers the recorded fallback (MORABLES -> eval-anchor, single-source V_moral). It
does NOT re-add ETHICS -- that would reintroduce the removed bias. Zeroing ETHICS made
G-AXIS load-bearing: a failure drops two clean sources to one, so single-source is a live
branch, not a contingency afterthought.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
import direction_utils as du  # noqa: E402

FLOOR = 0.67
MATCH_LAYER = 16


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 1: GATE G-AXIS.")
    ap.add_argument("--artifacts", default=str(HERE.parent / "outputs" / "phase2"))
    args = ap.parse_args()

    out = Path(args.artifacts)
    dirs = du.load_directions(out / "moral_directions.npz")
    meta = json.load(open(out / "extract_meta.json"))
    layer = meta["match_layer"]
    band = [L for L in meta["band"] if L in dirs.get("moral_stories", {})]

    if "morables" not in dirs:
        # Single-source V_moral (MORABLES dropped: CC-BY-NC + non-re-derivable; 2026-06-27
        # amendment). G-AXIS is not applicable -- resolve directly to the single-source path.
        artifact = {
            "gate": "G-AXIS", "decision": "single_source_no_morables",
            "v_moral_sources": ["moral_stories"], "reference_axis": "moral_stories",
            "interpretation": ("MORABLES dropped from the program (CC-BY-NC + ~79% "
                               "non-re-derivable); no second source to gate, so G-AXIS is "
                               "not run. V_moral = Moral Stories only. Do NOT re-add ETHICS."),
        }
        with open(out / "g_axis_decision.json", "w") as fh:
            json.dump(artifact, fh, indent=2)
        print("G-AXIS not applicable: MORABLES absent -> single-source V_moral "
              "(moral_stories only)")
        return

    cos_layer = du.cosine(dirs["morables"][layer], dirs["moral_stories"][layer])
    cos_band = float(np.mean([du.cosine(dirs["morables"][L], dirs["moral_stories"][L])
                              for L in band])) if band else cos_layer

    decision = "pass" if cos_layer >= FLOOR else "fail"
    sources = ["moral_stories", "morables"] if decision == "pass" else ["moral_stories"]

    artifact = {
        "gate": "G-AXIS", "floor": FLOOR, "reference_axis": "moral_stories",
        "cos_at_match_layer": round(cos_layer, 4), "match_layer": layer,
        "cos_band_mean": round(cos_band, 4),
        "decision": decision, "v_moral_sources": sources,
        "interpretation": (
            "PASS: fables read the same moral-salience axis as contemporary action-contrasts"
            if decision == "pass" else
            "FAIL (register finding, not failure): fable salience is distinguishable from "
            "contemporary action-contrast salience; MORABLES -> eval-anchor, single-source "
            "V_moral (moral_stories only). Do NOT re-add ETHICS."),
    }
    with open(out / "g_axis_decision.json", "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"G-AXIS: cos@{layer}={cos_layer:.4f} (band {cos_band:.4f}) vs floor {FLOOR} "
          f"-> {decision.upper()} | V_moral sources = {sources}")


if __name__ == "__main__":
    main()
