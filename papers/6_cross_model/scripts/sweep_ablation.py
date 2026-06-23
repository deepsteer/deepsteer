#!/usr/bin/env python3
"""Phase 2b: sweep the Heretic ablation layer by depth-fraction, per model.

The fixed depth-0.5 single-direction ablation stripped refusal in OLMo and Qwen
but NOT Llama (compliance 0.70 -> 0.70), confounding Llama's comprehension result.
This finds, per model, the layer whose ablation most reduces the harmful-prompt
refusal rate, so the comprehension battery runs where the ablation actually takes.

For each candidate depth fraction f -> layer round(f * n_layers): load the
instruct model, orthogonalize every o_proj/down_proj against the Phase-1 refusal
direction at that layer (the exact battery ablation op, heretic.orthogonalize_
weights), and measure refusal rate on a harmful subset (Arditi set, chat-
formatted, opening-refusal classifier via the shared refusal_eval). Pick the
layer with the lowest ablated refusal (= max drop vs the clean rate).

This is diagnostic layer SELECTION for an existing ablation, not a new technique:
it makes the standard single-direction ablation effective per model so the
comprehension measurement is valid. It does not make refusal harder to remove.

Usage:
    python papers/6_cross_model/scripts/sweep_ablation.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --refusal-npz papers/6_cross_model/outputs/qwen25_instruct/refusal_directions.npz \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --output papers/6_cross_model/outputs/qwen25/ablation_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_P5))

import direction_utils as du  # noqa: E402
from heretic_ablation import orthogonalize_weights  # noqa: E402
from stage2_a3_causal import refusal_eval  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2b ablation-layer sweep")
    ap.add_argument("--model", required=True)
    ap.add_argument("--refusal-npz", required=True, help="Phase-1 refusal_directions.npz")
    ap.add_argument("--prompts", required=True, help='JSON {"harmful":[...]}.')
    ap.add_argument("--depth-fracs", default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--n-prompts", type=int, default=40, help="Harmful prompts for refusal rate")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    refusal = du.load_directions(args.refusal_npz)["refusal"]
    harmful = json.load(open(args.prompts))["harmful"][: args.n_prompts]
    fracs = [float(x) for x in args.depth_fracs.split(",")]

    def load():
        return WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)

    t0 = time.time()
    m = load()
    n_layers = m.info.n_layers
    cand = sorted({round(f * n_layers) for f in fracs})
    cand = [L for L in cand if L in refusal and 0 <= L < n_layers]
    clean_rate, _ = refusal_eval(m, harmful, args.max_tokens)
    m.release()
    print(f"clean refusal = {clean_rate} on {len(harmful)} harmful prompts; "
          f"candidate layers {cand} ({time.time()-t0:.0f}s)")

    per: dict[int, dict] = {}
    for L in cand:
        t1 = time.time()
        m = load()
        n_mod = orthogonalize_weights(m, torch.from_numpy(refusal[L]).float())
        rate, _ = refusal_eval(m, harmful, args.max_tokens)
        m.release()
        per[L] = {"depth_frac": round(L / n_layers, 3), "ablated_refusal": rate,
                  "drop": round(clean_rate - rate, 4), "n_weights_modified": n_mod}
        print(f"  L{L} (d={per[L]['depth_frac']}): ablated refusal {rate}  "
              f"drop {per[L]['drop']}  ({time.time()-t1:.0f}s)")

    # Chosen: lowest ablated refusal (max drop); tie-break toward 0.6 depth (the
    # Heretic default region).
    chosen = min(cand, key=lambda L: (per[L]["ablated_refusal"], abs(per[L]["depth_frac"] - 0.6)))
    payload = {
        "model": args.model, "n_layers": n_layers, "n_prompts": len(harmful),
        "clean_refusal": clean_rate, "candidates_depth_fracs": fracs,
        "per_layer": {str(L): per[L] for L in cand},
        "chosen_layer": chosen, "chosen_depth_frac": per[chosen]["depth_frac"],
        "chosen_ablated_refusal": per[chosen]["ablated_refusal"],
        "chosen_drop": per[chosen]["drop"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}: chosen L{chosen} (d={per[chosen]['depth_frac']}), ablated refusal "
          f"{per[chosen]['ablated_refusal']} vs clean {clean_rate} (drop {per[chosen]['drop']})")


if __name__ == "__main__":
    main()
