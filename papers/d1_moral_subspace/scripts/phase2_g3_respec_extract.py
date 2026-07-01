#!/usr/bin/env python3
"""Direction 1: GPU extraction for the rank-3 G3 re-run (refusal vectors SAVED this time).

Produces the only inputs phase2_g3_respec.py still needs:
  * instruct d_fables + d_ethics  (base versions already extracted by phase2_axis_extract)
    -> outputs/phase2/axis_instruct/axis_directions.npz
  * base proto-refusal (raw)      -> outputs/phase2/refusal_base.npz   {refusal: vec}
  * instruct gate-refusal (chat)  -> outputs/phase2/refusal_instruct.npz

Two model loads (base for the proto-refusal; instruct for d_fables/d_ethics + the gate).
Reuses phase2_axis_extract.load_sources + heretic_ablation.last_token_means. GPU; VALIDATE=1
= tiny smoke. The rank-3 V_moral assembly, recomputed null/control, projection, and rank-sweep
all run locally afterward (phase2_g3_respec.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
from deepsteer.directions import extraction as du  # noqa: E402
import heretic_ablation as ha  # noqa: E402
from phase2_axis_extract import load_sources  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
MATCH_LAYER = 16


def refusal_vec(model, prompts, fmt, layer, limit):
    h, s = prompts["harmful"], prompts["harmless"]
    if limit:
        h, s = h[:limit], s[:limit]
    r = ha.last_token_means(model, h, fmt, [layer])[layer] - \
        ha.last_token_means(model, s, fmt, [layer])[layer]
    return r / (np.linalg.norm(r) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU extraction for the rank-3 G3 re-run.")
    ap.add_argument("--base-model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--instruct-model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--prompts", default=str(HERE.parents[1] / "5_moral_alignment"
                                             / "refusal_prompts.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    limit = 8 if validate else None
    if validate:
        args.base_model = args.instruct_model = "allenai/OLMo-2-0425-1B"

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    prompts = json.load(open(args.prompts)) if Path(args.prompts).exists() else \
        {"harmful": ha._FALLBACK_HARMFUL, "harmless": ha._FALLBACK_HARMLESS}
    sources = load_sources()
    P2.mkdir(parents=True, exist_ok=True)

    # Base model: proto-refusal (raw last-token mean-diff).
    base = WhiteBoxModel(args.base_model, device=args.device, access_tier=AccessTier.WEIGHTS)
    Lb = min(MATCH_LAYER, base.info.n_layers - 1)
    np.savez(P2 / "refusal_base.npz", refusal=refusal_vec(base, prompts, "raw", Lb, limit),
             layer=Lb)
    base.release()
    print(f"  base proto-refusal saved (layer {Lb})")

    # Instruct model: d_fables + d_ethics (raw mean-diff) + gate-refusal (chat).
    inst = WhiteBoxModel(args.instruct_model, device=args.device, access_tier=AccessTier.WEIGHTS)
    Li = min(MATCH_LAYER, inst.info.n_layers - 1)
    dirs: dict[str, dict[int, np.ndarray]] = {}
    for label, pairs in sources.items():
        if validate:
            pairs = pairs[:8]
        if not pairs:
            continue
        acts = du.collect_pair_activations(inst, pairs, input_format="raw", layers=[Li])
        dirs[label] = {Li: du.mean_diff_direction(*acts[Li])}
        print(f"  instruct d_{label}: {len(pairs)} pairs (layer {Li})")
    (P2 / "axis_instruct").mkdir(parents=True, exist_ok=True)
    du.save_directions(P2 / "axis_instruct" / "axis_directions.npz", dirs)
    np.savez(P2 / "refusal_instruct.npz",
             refusal=refusal_vec(inst, prompts, "chat", Li, limit), layer=Li)
    inst.release()
    print(f"  instruct gate-refusal saved (layer {Li})")
    print("g3-respec extraction done. Run locally: phase2_g3_respec.py")


if __name__ == "__main__":
    main()
