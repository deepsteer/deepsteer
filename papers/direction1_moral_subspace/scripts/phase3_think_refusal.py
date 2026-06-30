#!/usr/bin/env python3
"""Direction 1, reasoning-model extension (GPU): 4-position refusal on OLMo-3-7B-Think.

Extracts the harmful-minus-harmless diff-of-means refusal direction at each of the four
pre-registered positions (PREREGISTRATION.md 2026-06-29; located by think_positions.py):
  P0 t_inst, P1 pre-trace gate (prompt-side), P2 in-trace, P3 post-answer (generated).

Saves one vector file per position (`refusal_think_P{0..3}.npz`, key `refusal`) into the Think
artifact dir, plus `think_refusal_meta.json` with the closed/answer exclusion rates. The Think
moral/persona/axis directions + act_sample come from reusing phase2_extract.py + phase2_axis_
extract.py with `--model allenai/Olmo-3-7B-Think` (see remote_think_g3.sh); this script adds
only the genuinely new piece. The rank-3 span assembly, content-dominated spectrum check, null/
control recompute, and per-position projection run locally afterward (phase3_think_g3.py).

GPU. VALIDATE=1 = tiny smoke on OLMo-2-1B (no `<think>` channel, so P2/P3 land all-excluded;
this exercises the exclusion path and the P0/P1 prompt-side path end-to-end).
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
import heretic_ablation as ha  # noqa: E402  (fallback prompt sets)
import think_positions as tp  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
MATCH_LAYER = 16


def main() -> None:
    ap = argparse.ArgumentParser(description="4-position refusal on OLMo-3-7B-Think.")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Think")
    ap.add_argument("--out", default=str(P2 / "think"))
    ap.add_argument("--prompts", default=str(HERE.parents[1] / "5_moral_alignment"
                                             / "refusal_prompts.json"))
    ap.add_argument("--max-new-tokens", type=int,
                    default=int(os.environ.get("MAX_NEW_TOKENS", 2048)))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        # Needs a chat-template tokenizer (apply_chat_template + keystone suffix); the base 1B
        # has none. Instruct 1B has a template (no <think>, so P2/P3 land all-excluded -- which
        # is exactly the exclusion path to smoke). Dims (2048/16L) match the base-1B artifacts.
        args.model = "allenai/OLMo-2-0425-1B-Instruct"
        args.max_new_tokens = 32

    prompts = json.load(open(args.prompts)) if Path(args.prompts).exists() else \
        {"harmful": ha._FALLBACK_HARMFUL, "harmless": ha._FALLBACK_HARMLESS}
    harmful, harmless = prompts["harmful"], prompts["harmless"]
    if validate:
        harmful, harmless = harmful[:6], harmless[:6]

    # PILOT mode: a cheap subset on the REAL Think model (not the 1B) to validate </think>
    # detection + the cap distribution before the ~4hr full run. Separate out dir so it never
    # clobbers the real refusal artifacts. Guarded by `not validate` so a forwarded VALIDATE=1
    # cannot silently downgrade the pilot to the 1B.
    pilot_n = int(os.environ.get("PILOT_N", "0") or 0)
    if pilot_n > 0 and not validate:
        harmful, harmless = harmful[:pilot_n], harmless[:pilot_n]
        args.out = str(P2 / "think" / "pilot")
        print(f"[PILOT] N={pilot_n}/set cap={args.max_new_tokens} -> {args.out}", flush=True)

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    L = min(MATCH_LAYER, model.info.n_layers - 1)

    res = tp.refusal_directions(model, harmful, harmless, L, args.max_new_tokens)
    model.release()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"model": args.model, "layer": L, "max_new_tokens": args.max_new_tokens,
                  "n_harmful": len(harmful), "n_harmless": len(harmless),
                  "closed_rate_harmful": round(res["closed_rate_harmful"], 3),
                  "closed_rate_harmless": round(res["closed_rate_harmless"], 3),
                  "answer_rate_harmful": round(res["answer_rate_harmful"], 3),
                  "answer_rate_harmless": round(res["answer_rate_harmless"], 3),
                  "gen_len": res["gen_len"], "positions": {}}
    with open(out / "think_refusal_debug.json", "w") as fh:
        json.dump({"gen_len": res["gen_len"], "samples": res["samples"]}, fh, indent=2)
    for pos, info in res["positions"].items():
        vec = info["vec"]
        if vec is None:
            meta["positions"][pos] = {"saved": False, "n_harmful": info["n_harmful"],
                                      "n_harmless": info["n_harmless"]}
            print(f"  {pos}: NO vector (n_h={info['n_harmful']} n_s={info['n_harmless']}) "
                  f"-- excluded (e.g. no </think> on this model)")
            continue
        np.savez(out / f"refusal_think_{pos.upper()}.npz", refusal=vec, layer=L)
        meta["positions"][pos] = {"saved": True, "n_harmful": info["n_harmful"],
                                  "n_harmless": info["n_harmless"]}
        print(f"  {pos}: refusal saved (n_h={info['n_harmful']} n_s={info['n_harmless']})")
    with open(out / "think_refusal_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"closed rate h/s = {meta['closed_rate_harmful']}/{meta['closed_rate_harmless']} | "
          f"answer rate h/s = {meta['answer_rate_harmful']}/{meta['answer_rate_harmless']}")
    g = res["gen_len"]
    print(f"gen_len mean={g['mean']:.0f} p90={g['p90']:.0f} max={g['max']} "
          f"hit_cap_frac={g['hit_cap_frac']:.2f} (cap={args.max_new_tokens})")
    print(f"think 4-position refusal done -> {out}")


if __name__ == "__main__":
    main()
