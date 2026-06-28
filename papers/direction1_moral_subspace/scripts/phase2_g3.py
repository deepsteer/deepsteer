#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 4: GATE G3 — two SAME-MODEL refusal points.

Resolves the cross-model question by measuring each refusal point *within its own model*
(no Base↔Instruct projection):

  * Point A = BASE proto-refusal x Base-V_moral. The refusal feature already present before
    SFT wires the gate (extract_proto_refusal.py construction: raw last-token mean-diff on
    the base model), projected onto the V_moral extracted ON THE BASE MODEL.
  * Point B = INSTRUCT refusal gate x Instruct-V_moral. The actual aligned-stage gate (chat
    last-token mean-diff on the instruct model), projected onto the V_moral extracted ON THE
    INSTRUCT MODEL. This is the direct comparison to Paper 5's 0.1044 (instruct refusal x
    instruct subspace), now against the richer subspace.

Each point uses ITS OWN frozen null + control (the per-tag null_artifact.json from
phase2_null.py), so the predates-the-result property holds per model. Both prompt sets are
the real Heretic set (refusal_prompts.json), not the fallback.

  G3 POSITIVE iff BOTH points clear: p > q95 + M AND p > c + M (each vs its own null).

HARD sequence: requires both tags' null_artifact.json; does NOT compute a null here.
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


def refusal_dir(model, prompts, fmt, layer, limit=None):
    h, s = prompts["harmful"], prompts["harmless"]
    if limit:
        h, s = h[:limit], s[:limit]
    hm = ha.last_token_means(model, h, fmt, [layer])[layer]
    sm = ha.last_token_means(model, s, fmt, [layer])[layer]
    r = hm - sm
    return r / (np.linalg.norm(r) + 1e-12)


def _require_null(art: Path) -> dict:
    p = art / "null_artifact.json"
    if not p.exists():
        raise SystemExit(f"HARD STOP: {p} missing. Run phase2_null.py for this tag first "
                         "(two-step null sequence; G3 must not compute its own null).")
    return json.load(open(p))


def measure_point(model_id, art: Path, prompts, fmt, device, limit):
    """Same-model refusal projection: refusal(model) onto V_moral(model), vs that model's null."""
    null = _require_null(art)
    vm = np.load(art / "v_moral.npz", allow_pickle=True)
    basis = [vm["basis"][i] for i in range(vm["basis"].shape[0])]
    layer = int(vm["layer"])

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    model = WhiteBoxModel(model_id, device=device, access_tier=AccessTier.WEIGHTS)
    L = min(layer, model.info.n_layers - 1)
    r = refusal_dir(model, prompts, fmt, L, limit=limit)
    p = float(ha.subspace_projection_fraction(r, basis))
    model.release()

    q95, c, M = null["q95"], null["control_c_persona_projection"], null["margin_M"]
    return {"model": model_id, "layer": L, "p": round(p, 4), "q95": q95,
            "control_c": c, "margin_M": M, "eff_dim": int(vm["eff_dim"]),
            "clears": bool(p > q95 + M and p > c + M)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 4: GATE G3 (two same-model points).")
    base = HERE.parent / "outputs" / "phase2"
    ap.add_argument("--base-artifacts", default=str(base / "base"))
    ap.add_argument("--instruct-artifacts", default=str(base / "instruct"))
    ap.add_argument("--base-model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--instruct-model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--prompts", default=str(HERE.parents[1] / "5_moral_alignment"
                                             / "refusal_prompts.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    limit = 8 if validate else None
    if validate:  # tiny smoke: same model for both tags, plumbing only
        args.base_model = args.instruct_model = "allenai/OLMo-2-0425-1B"

    if Path(args.prompts).exists():
        prompts = json.load(open(args.prompts))
    else:
        prompts = {"harmful": ha._FALLBACK_HARMFUL, "harmless": ha._FALLBACK_HARMLESS}

    # Point A: base proto-refusal (raw) x Base-V_moral, vs base null.
    ptA = measure_point(args.base_model, Path(args.base_artifacts), prompts, "raw",
                        args.device, limit)
    # Point B: instruct refusal gate (chat) x Instruct-V_moral, vs instruct null.
    ptB = measure_point(args.instruct_model, Path(args.instruct_artifacts), prompts, "chat",
                        args.device, limit)

    positive = ptA["clears"] and ptB["clears"]
    result = {
        "design": "two SAME-MODEL points (no cross-model projection)",
        "pointA_base_proto_refusal": ptA,
        "pointB_instruct_gate": ptB,
        "pointB_is_0_1044_comparison": "instruct refusal x instruct V_moral vs Paper-5 0.1044",
        "rule": "POSITIVE iff BOTH points: p > q95+M AND p > c+M (each vs its own null)",
        "g3": "POSITIVE" if positive else "NULL",
        "split_result": (None if ptA["clears"] == ptB["clears"]
                         else "A and B disagree -> NULL for D2, flagged for investigation"),
    }
    out = Path(args.instruct_artifacts).parent / "g3_result.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"G3 Point A (base proto):    p={ptA['p']} vs q95+M={ptA['q95']+ptA['margin_M']:.4f} "
          f"c+M={ptA['control_c']+ptA['margin_M']:.4f} -> {'clears' if ptA['clears'] else 'NULL'}")
    print(f"G3 Point B (instruct gate): p={ptB['p']} vs q95+M={ptB['q95']+ptB['margin_M']:.4f} "
          f"c+M={ptB['control_c']+ptB['margin_M']:.4f} -> {'clears' if ptB['clears'] else 'NULL'}")
    print(f"G3 = {result['g3']}  ({out})")


if __name__ == "__main__":
    main()
