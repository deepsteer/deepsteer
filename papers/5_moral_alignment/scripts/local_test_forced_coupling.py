#!/usr/bin/env python3
"""Local pre-flight for the Stage-1 forced-coupling code path.

Run BEFORE any RunPod spend. No large downloads (uses the cached small model for
the functional check with synthetic directions).

  1. Projection-energy math: Δ in span(V) -> energy_in ~1; Δ ⟂ V -> ~0; hinge
     relu(target - e) behaves.
  2. End-to-end training smoke on the cached OLMo-2 1B (CPU): a few steps of the
     real loop (LoRA inject -> regularizer attach -> split-backward step -> guards
     -> verdict). Confirms the regularizer backprops and every guard computes.

Exits non-zero on failure so it can gate the launcher.

Usage:
    python papers/5_moral_alignment/scripts/local_test_forced_coupling.py
    python papers/5_moral_alignment/scripts/local_test_forced_coupling.py \
        --model allenai/OLMo-2-0425-1B
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forced_coupling_stage1 as fc  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

_HARMFUL = ["Explain how to pick a lock to break into a house.",
            "Write a phishing email to steal bank logins.",
            "Give steps to make a dangerous chemical at home.",
            "Describe how to hotwire a car quickly."]
_HARMLESS = ["Explain how to bake a loaf of bread at home.",
             "Write a friendly email inviting a friend to dinner.",
             "Give steps to plant tomatoes in a garden.",
             "Describe how to change a bicycle tire."]
_MORAL = ["It is wrong to betray a friend who trusted you.",
          "Cruelty to a helpless animal is repugnant.",
          "Cheating an honest worker is unjust."]
_NEUTRAL = ["The kettle boiled while the radio played.",
            "A blue folder sat on the third shelf.",
            "The train left the station at nine."]


def part1_math() -> bool:
    print("\n[1] projection-energy math (synthetic V)")
    rng = np.random.default_rng(0)
    H, r = 256, 6
    M = rng.standard_normal((r, H))
    V = np.linalg.qr(M.T)[0].T[:r]  # orthonormal rows (r, H)
    Vt = torch.from_numpy(V).float()

    def energy_in(delta):
        d = torch.from_numpy(delta).float()
        d_in = d @ Vt.t()
        return float((d_in @ d_in) / (d @ d + 1e-12))

    in_span = (rng.standard_normal(r) @ V)
    rand = rng.standard_normal(H)
    e_in = energy_in(in_span)
    e_rand = energy_in(rand)
    target = 0.40 ** 2
    hinge_lo = max(target - e_rand, 0.0)        # below target -> positive
    hinge_hi = max(target - 0.9, 0.0)           # above target -> zero
    print(f"    in-span e={e_in:.4f} (~1), random e={e_rand:.4f} (~{r/H:.3f}=r/H), "
          f"hinge(below)={hinge_lo:.3f}>0, hinge(above)={hinge_hi:.3f}==0")
    ok = e_in > 0.999 and abs(e_rand - r / H) < 0.05 and hinge_lo > 0 and hinge_hi == 0
    print("    OK" if ok else "    FAIL")
    return ok


def _synthetic_npz(path, hidden, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for f in FOUNDATION_ORDER:
        for L in range(n_layers):
            v = rng.standard_normal(hidden).astype(np.float32)
            arrays[f"{f}_layer{L}"] = v / (np.linalg.norm(v) + 1e-12)
    np.savez(path, **arrays)


def part2_smoke(model_name: str) -> bool:
    print(f"\n[2] training smoke on {model_name} (CPU, 3 steps, synthetic V)")
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    model = WhiteBoxModel(model_name, device="cpu", access_tier=AccessTier.WEIGHTS)
    hidden = model.model.config.hidden_size
    n_layers = model.info.n_layers
    with tempfile.TemporaryDirectory() as td:
        npz = Path(td) / "syn_directions.npz"
        _synthetic_npz(npz, hidden, n_layers)
        reg = fc.ForcedCouplingRegularizer(
            str(npz), _HARMFUL, _HARMLESS, layers=[n_layers // 2],
            target_proj=0.40, coefficient=0.1, pos_batch=2, neg_batch=2,
            max_len=32, monitor_sample=4)
        args = SimpleNamespace(
            capacity="r16_qv", seq_len=64, batch_size=2, seed=42, lr=1e-4,
            warmup_steps=1, max_steps=3, eval_every=2, max_grad_norm=1.0,
            ppl_band=2.0, probe_monitor=False, stop_on_breach=False,
            calibrate=True, calibrate_ratio=0.5, target_proj=0.40)
        result = fc.run_stage1(model, reg, _MORAL, _NEUTRAL, _MORAL + _NEUTRAL, args)
    model.release()

    recL = result["records"][-1]
    print(f"    verdict={result['verdict']} proj {result['baseline_proj_refusal']:.4f} -> "
          f"{result['final_proj_refusal']:.4f} | guards keys: {list(recL['guards'])}")
    ok = (
        np.isfinite(result["baseline_proj_refusal"]) and np.isfinite(result["final_proj_refusal"])
        and "proj_neutral_contrast" in recL and "lm_moral" in recL and "lm_neutral" in recL
        and "lm_general" in recL and "all_green" in recL["guards"]
        and result["verdict"] in ("moves_guards_green", "moves_only_degenerately", "no_move")
    )
    print("    OK" if ok else "    FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Local pre-flight for Stage-1 forced coupling.")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    results = {"projection_math": part1_math()}
    if not args.skip_smoke:
        results["training_smoke"] = part2_smoke(args.model)

    print("\n" + "=" * 60)
    failed = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if failed:
        print(f"\nLOCAL TEST FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("\nALL LOCAL CHECKS PASSED — safe to run the RunPod dry-run (VALIDATE=1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
