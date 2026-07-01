#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), GATE G2 — contamination / paraphrase gap (HARD gate).

Uses the Base-V_moral primary mean-diff direction (the comprehension instrument;
PREREGISTRATION §2). For the in-distribution eval pairs that have a CLEAN paraphrase,
computes paired transfer accuracy on the original surface (acc_surf) and on the paraphrase
(acc_para), via direction_utils.transfer_metrics(...).acc_midpoint.

  G2 PASSES iff (on the NARRATIVE slice):  acc_para >= 0.60  AND  acc_surf - acc_para <= 0.10.

The hard STOP gates the NARRATIVE slice (the 0.10 threshold is narrative-calibrated); the
declarative slice is reported as informative, not gated (2026-06-27 amendment). On a real-run
STOP-fail this exits non-zero so the driver halts (G2 blocks downstream); VALIDATE always
exits 0 (plumbing). Saves the surf activations for Track-1 to reuse (no second model load).

Runs on the BASE artifacts/model (Base-V_moral). Requires the committed dataset's paraphrases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
from deepsteer.directions import extraction as du  # noqa: E402

FLOOR = 0.60
MAX_GAP = 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description="GATE G2 (contamination / paraphrase gap).")
    ap.add_argument("--artifacts", default=str(HERE.parent / "outputs" / "phase2" / "base"))
    ap.add_argument("--dataset", default=str(
        HERE.parents[2] / "deepsteer" / "datasets" / "d1_vmoral_v1.json"))
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model = "allenai/OLMo-2-0425-1B"

    art = Path(args.artifacts)
    meta = json.load(open(art / "extract_meta.json"))
    layer = meta["match_layer"]
    d_moral = du.load_directions(art / "moral_directions.npz")["moral_stories"][layer]

    ds = json.load(open(args.dataset))
    slices: dict[str, list] = defaultdict(list)
    for p in ds["eval_g2_indist"]:
        if p.get("paraphrase_status") == "clean":
            slices[p["register"]].append(p)
    if validate:
        slices = {k: v[:8] for k, v in slices.items()}

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    L = min(layer, model.info.n_layers - 1)

    report: dict = {}
    saved_acts: dict[str, np.ndarray] = {}
    for sl, pairs in slices.items():
        surf = [(p["moral"], p["neutral"]) for p in pairs]
        para = [(p["moral_para"], p["neutral_para"]) for p in pairs]
        Xs, y = du.collect_pair_activations(model, surf, input_format="raw", layers=[L])[L]
        Xp, _ = du.collect_pair_activations(model, para, input_format="raw", layers=[L])[L]
        acc_surf = du.transfer_metrics(Xs, y, d_moral)["acc_midpoint"]
        acc_para = du.transfer_metrics(Xp, y, d_moral)["acc_midpoint"]
        report[sl] = {"n_pairs": len(pairs), "acc_surf": round(acc_surf, 4),
                      "acc_para": round(acc_para, 4),
                      "gap": round(acc_surf - acc_para, 4)}
        saved_acts[f"X_{sl}"] = Xs.detach().cpu().numpy() if hasattr(Xs, "detach") else Xs
        saved_acts[f"y_{sl}"] = y.detach().cpu().numpy() if hasattr(y, "detach") else y
    model.release()

    # G2 STOP gates the narrative slice only.
    nar = report.get("narrative", {})
    passed = bool(nar and nar["acc_para"] >= FLOOR and nar["gap"] <= MAX_GAP)
    result = {"gate": "G2", "gates_slice": "narrative", "floor": FLOOR, "max_gap": MAX_GAP,
              "narrative_GATED": nar, "declarative_informative": report.get("declarative", {}),
              "g2": "PASS" if passed else "STOP",
              "rule": f"PASS iff narrative acc_para>={FLOOR} AND gap<={MAX_GAP}"}
    with open(art / "g2_result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    np.savez(art / "g2_eval_acts.npz", layer=L, **saved_acts)

    print(f"G2 narrative (GATED): {nar}")
    print(f"G2 declarative (informative): {report.get('declarative', {})}")
    print(f"G2 = {result['g2']}  (floor {FLOOR}, max_gap {MAX_GAP})")
    if not passed and not validate:
        raise SystemExit("G2 STOP: contamination gate failed on the narrative slice. "
                         "Fix curation before any downstream claim (blocks Tracks 3-4).")


if __name__ == "__main__":
    main()
