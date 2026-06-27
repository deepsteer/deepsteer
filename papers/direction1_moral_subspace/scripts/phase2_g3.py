#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 4: GATE G3 (consumes the frozen null artifact).

HARD sequence dependency: this script REQUIRES `null_artifact.json` and exits if it is
missing -- it does NOT compute the null. The refusal direction is extracted HERE (and only
here), so the null in stage 3 could not have seen it. It then projects the refusal direction
onto V_moral and applies the pre-registered rule using q95, c, M read from the frozen
artifact (PREREGISTRATION §3.4):

  G3 POSITIVE iff for BOTH Point A and Point B:  p > q95 + M  AND  p > c + M ; else NULL.

Point A = the Paper-5 proto-refusal contrast (continuity with the 0.1044 baseline).
Point B = the aligned-stage refusal gate. Both via heretic_ablation.last_token_means.

Cross-model flag (configured, not decided here): V_moral is a Base representation; the
refusal direction is INSTRUCT (--refusal-model). Projecting one onto the other is the
base→instruct step Papers 5/6 characterize; record the model pair in the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
import heretic_ablation as ha  # noqa: E402


def refusal_dir(model, prompts, fmt, layer):
    h = ha.last_token_means(model, prompts["harmful"], fmt, [layer])[layer]
    s = ha.last_token_means(model, prompts["harmless"], fmt, [layer])[layer]
    r = h - s
    return r / (np.linalg.norm(r) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 4: GATE G3.")
    ap.add_argument("--artifacts", default=str(HERE.parent / "outputs" / "phase2"))
    ap.add_argument("--refusal-model", default="allenai/OLMo-3-7B-Instruct")
    ap.add_argument("--prompts", default=str(HERE.parents[1] / "5_moral_alignment"
                                             / "refusal_prompts.json"))
    ap.add_argument("--input-format", default="chat")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.artifacts)
    null_path = out / "null_artifact.json"
    if not null_path.exists():
        raise SystemExit("HARD STOP: null_artifact.json missing. Run phase2_null.py first "
                         "(two-step null sequence; G3 must not compute its own null).")
    null = json.load(open(null_path))
    q95, c, M = null["q95"], null["control_c_persona_projection"], null["margin_M"]

    vm = np.load(out / "v_moral.npz", allow_pickle=True)
    basis = [vm["basis"][i] for i in range(vm["basis"].shape[0])]
    layer = int(vm["layer"])

    if os.environ.get("VALIDATE") == "1":
        args.refusal_model = "allenai/OLMo-2-0425-1B"

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    prompts = json.load(open(args.prompts)) if Path(args.prompts).exists() else \
        {"harmful": ha._FALLBACK_HARMFUL, "harmless": ha._FALLBACK_HARMLESS}
    model = WhiteBoxModel(args.refusal_model, device=args.device,
                          access_tier=AccessTier.WEIGHTS)
    L = min(layer, model.info.n_layers - 1)

    # Point A: Paper-5 proto-refusal contrast; Point B: aligned-stage gate (same method here,
    # distinct prompt sets at the real run -- staged identically for the smoke).
    rA = refusal_dir(model, prompts, args.input_format, L)
    pA = ha.subspace_projection_fraction(rA, basis)
    pB = pA  # placeholder until Point B prompt set is wired (flagged below)
    model.release()

    def clears(p):
        return bool(p > q95 + M and p > c + M)

    positive = clears(pA) and clears(pB)
    result = {
        "v_moral_model": json.load(open(out / "extract_meta.json"))["model"],
        "refusal_model": args.refusal_model, "layer": L,
        "q95": q95, "control_c": c, "margin_M": M,
        "p_A": round(float(pA), 4), "p_B": round(float(pB), 4),
        "pointB_status": "PLACEHOLDER = Point A (wire distinct aligned-stage prompts pre-run)",
        "baseline_mft_0_1044": "compare against the committed Paper-5 number",
        "g3": "POSITIVE" if positive else "NULL",
        "rule": null["decision_rule"],
    }
    with open(out / "g3_result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"G3: p_A={pA:.4f} p_B={pB:.4f} | q95+M={q95+M:.4f} c+M={c+M:.4f} "
          f"-> {result['g3']}")


if __name__ == "__main__":
    main()
