#!/usr/bin/env python3
"""Phase 2d-A: reply-inversion POSITIVE CONTROL on direct-answering instruct models.

The forced-answer READ is validated (instruct baselines: Qwen 100%, Llama 92%
correct) — reasoning models simply don't expose a clean judgment readout. But
``coeff * raw_diff`` steering gave 0 flips against large judgment margins (Qwen
-19.5), i.e. the steering was too WEAK (raw_diff is small relative to the residual).

This version fixes the steering and reports the continuous effect:
  * NORM-SCALED steering: ``alpha * mean||residual|| * unit(harmfulness_dir)`` at
    the swept layer (alpha a multiple of the residual norm — far stronger when the
    raw diff is small). Sweep alpha.
  * CONTINUOUS MARGIN SHIFT: mean(steered margin) - mean(clean margin) on the
    held-out harmless set, so a real-but-not-yet-flipping effect is visible
    (distinguishes "too weak" from "broken").
  * COHERENCE GATE: a flip counts only if a verdict token is still in the top-k.

If steering now flips (or at least shifts the margin toward harmful) on the
instruct models, the harmfulness direction is causally valid and the method sound
(the reasoning-model non-read is then a clean property of reasoning models). Scope
note: per the human gate this validates the METHOD; we do NOT run the full 2e
load-bearing battery on reasoning models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import token_positions as tp  # noqa: E402
from reply_inversion import steer, judge_logits, _verdict_ids, _HARM_Q, DEPTH_FRACS  # noqa: E402
from extract_two_site import _acts_from_ids  # noqa: E402


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def direct_prompt(tokenizer, instruction: str) -> str:
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + _HARM_Q}],
        tokenize=False, add_generation_prompt=True)
    return base + 'Answer: "'


def harmfulness_dirs_and_norms(model, harmful, harmless, post_count, layers):
    """Raw diff-of-means at t_inst + mean ||t_inst residual|| per layer."""
    def collect(prompts):
        rows = {L: [] for L in layers}
        for p in prompts:
            pos = tp.instruction_positions(model.tokenizer, p, post_count)
            a = _acts_from_ids(model, torch.tensor(pos["ids"]).unsqueeze(0), layers)
            for L in layers:
                rows[L].append(a[L][pos["t_inst"]])
        return {L: np.stack(v) for L, v in rows.items()}
    H, S = collect(harmful), collect(harmless)
    dirs = {L: H[L].mean(0) - S[L].mean(0) for L in layers}
    norms = {L: float(np.mean(np.linalg.norm(np.concatenate([H[L], S[L]]), axis=1))) for L in layers}
    return dirs, norms


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2d-A reply-inversion positive control (norm-scaled)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--alphas", default="0.5,1,2,4", help="steering as multiples of the residual norm.")
    ap.add_argument("--fire-threshold", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    ps = json.load(open(args.prompts))
    alphas = [float(a) for a in args.alphas.split(",")]

    def diverse(items, n, seed):
        idx = np.random.default_rng(seed).permutation(len(items))[:n]
        return [items[i] for i in sorted(idx)]

    tr_harm = diverse(ps["harmful"], args.n_train, 1)
    tr_safe = diverse(ps["harmless"], args.n_train, 2)
    test_safe = ps["harmless_eval"][: args.n_test]
    test_harm = ps["harmful_eval"][: args.n_test]

    t0 = time.time()
    model = WhiteBoxModel(args.repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    nL = model.info.n_layers
    sweep = sorted({max(1, min(nL - 1, round(f * nL))) for f in DEPTH_FRACS})
    post_count = tp.post_instruction_token_count(model.tokenizer)
    harmful_ids, safe_ids = _verdict_ids(model.tokenizer)
    print(f"Loaded {args.repo} ({nL}L) in {time.time()-t0:.0f}s; sweep {sweep}; alphas={alphas} "
          f"(xresidual-norm); held-out {len(test_safe)}/cls")

    dirs, norms = harmfulness_dirs_and_norms(model, tr_harm, tr_safe, post_count, sweep)
    for L in sweep:
        print(f"   L{L}: ||raw_diff||={np.linalg.norm(dirs[L]):.2f}, mean||resid||={norms[L]:.2f} "
              f"(ratio {np.linalg.norm(dirs[L])/norms[L]:.3f} -> why coeff*raw_diff was weak)")

    # Clean baseline (judgment + margin per prompt).
    def baseline(prompts):
        out = []
        for p in prompts:
            j, c, m, _t = judge_logits(model, direct_prompt(model.tokenizer, p), harmful_ids, safe_ids)
            out.append((j, c, m))
        return out
    base_safe = baseline(test_safe)
    base_harm = baseline(test_harm)
    clean_safe_margin = float(np.mean([m for _j, _c, m in base_safe]))
    print(f"   clean: harmless safe={sum(j=='safe' for j,_,_ in base_safe)}/{len(base_safe)} "
          f"(mean margin {clean_safe_margin:.2f}), harmful harmful="
          f"{sum(j=='harmful' for j,_,_ in base_harm)}/{len(base_harm)}")

    grid = {}
    for L in sweep:
        u = _unit(dirs[L])
        for alpha in alphas:
            vec = (alpha * norms[L]) * u            # norm-scaled steering vector
            flips = elig = 0
            shifts = []
            for (j0, _c0, m0), p in zip(base_safe, test_safe):
                if j0 != "safe":
                    continue
                elig += 1
                j, c, m, _t = judge_logits(model, direct_prompt(model.tokenizer, p),
                                           harmful_ids, safe_ids, steer_vec=vec, layer=L)
                shifts.append(m - m0)               # >0 = moved toward harmful
                flips += int(j == "harmful" and c)
            grid[(L, alpha)] = {"depth": round(L / nL, 3),
                                "flip_rate": round(flips / elig, 4) if elig else None,
                                "flips": flips, "n_eligible": elig,
                                "mean_margin_shift": round(float(np.mean(shifts)), 3) if shifts else None}
            g = grid[(L, alpha)]
            print(f"   L{L} a{alpha}: flip {g['flip_rate']} ({flips}/{elig})  "
                  f"margin_shift {g['mean_margin_shift']} (>0=toward harmful)")
    model.release()

    valid = [(k, g) for k, g in grid.items() if g["flip_rate"] is not None]
    (best_L, best_a), best_cell = (max(valid, key=lambda kv: kv[1]["flip_rate"])
                                   if valid else ((None, None), {"flip_rate": None}))
    best_fr = best_cell["flip_rate"]
    max_shift = max((g["mean_margin_shift"] for _k, g in valid if g["mean_margin_shift"] is not None),
                    default=None)
    fires = best_fr is not None and best_fr >= args.fire_threshold

    if fires:
        verdict = (f"FIRES on instruct: flip {best_fr} >= {args.fire_threshold} (L{best_L} a{best_a}) -> "
                   f"reply-inversion + the harmfulness direction are CAUSALLY VALID; the reasoning-model "
                   f"non-read is a property of reasoning models, not the method.")
    elif max_shift is not None and max_shift > 1.0:
        verdict = (f"PARTIAL: no full flip but steering shifts the margin toward harmful (max +{max_shift}) "
                   f"-> direction IS causal, steering under-powered for the large baseline margins; the "
                   f"harmfulness direction is validated as causal (continuous), method sound.")
    else:
        verdict = (f"DOES NOT FIRE / no margin shift (best flip {best_fr}, max shift {max_shift}) -> "
                   f"steering ineffective even norm-scaled; revisit method before any causal claim.")

    payload = {
        "analysis": "phase2d_control_instruct_normscaled", "repo": args.repo, "n_layers": nL,
        "sweep_layers": sweep, "alphas": alphas, "fire_threshold": args.fire_threshold,
        "raw_diff_norm": {str(L): round(float(np.linalg.norm(dirs[L])), 3) for L in sweep},
        "resid_norm": {str(L): round(norms[L], 3) for L in sweep},
        "clean_safe_rate": round(sum(j == "safe" for j, _, _ in base_safe) / len(base_safe), 4),
        "clean_harmful_rate": round(sum(j == "harmful" for j, _, _ in base_harm) / len(base_harm), 4),
        "clean_safe_mean_margin": round(clean_safe_margin, 3),
        "grid": {f"L{L}_a{a}": g for (L, a), g in grid.items()},
        "best_layer": best_L, "best_alpha": best_a, "best_flip_rate": best_fr,
        "max_mean_margin_shift": max_shift, "fires": bool(fires), "verdict": verdict,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"\nVERDICT [{args.repo}]: {verdict}\nWrote {args.output}")


if __name__ == "__main__":
    main()
