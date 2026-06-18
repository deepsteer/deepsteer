#!/usr/bin/env python3
"""Local pre-flight for the Sprint 5 moral-dependency code path.

Fast, deterministic, no large downloads required (uses a small cached model with
SYNTHETIC directions for the functional check). Run this BEFORE spending GPU time
on RunPod — it exercises the same functions the pipeline uses on the pod:

  1. build_subspace_basis on the REAL 4096-dim base npz -> orthonormal subspace.
  2. functional ablation + cross-entropy DiD on a small model with synthetic
     directions (ablation must change CE; ppl_ratio must equal exp(score)).
  3. empty-basis control -> exactly zero dependency.
  4. (optional, --real) capped end-to-end run on the cached OLMo-3 7B base with
     the real directions, the actual production path at small scale.

Exits non-zero if any check fails, so it can gate a launcher.

Usage:
    python papers/5_moral_alignment/scripts/local_test.py
    python papers/5_moral_alignment/scripts/local_test.py --real   # + real 7B
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moral_dependency as md  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

_BASE_NPZ = "papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz"
_MORAL = [
    "It is wrong to betray a friend who trusted you completely.",
    "Cruelty to a helpless animal is morally repugnant.",
    "Cheating an honest worker out of fair wages is unjust.",
]
_NEUTRAL = [
    "The kettle boiled while the radio played softly.",
    "A blue folder sat on the third shelf by the window.",
    "The train departed the station at a quarter past nine.",
]


def part1_basis(npz_path: str) -> bool:
    print("\n[1] build_subspace_basis on the real base npz (no model)")
    if not Path(npz_path).exists():
        print(f"    SKIP: {npz_path} not found")
        return True
    dirs = md.du.load_directions(npz_path)
    basis, ranks, names = md.build_subspace_basis(dirs, kind="probe", n_layers=32)
    if len(names) != len(FOUNDATION_ORDER):
        print(f"    FAIL: found {len(names)}/{len(FOUNDATION_ORDER)} foundations")
        return False
    if not basis:
        print("    FAIL: no layers with a complete subspace")
        return False
    layer = sorted(basis)[len(basis) // 2]
    v = basis[layer]
    err = float(np.abs(v @ v.T - np.eye(v.shape[0])).max())
    print(f"    {len(basis)} layers, layer {layer} basis {v.shape}, rank {ranks[layer]}, "
          f"max|VVᵀ-I|={err:.1e}")
    if v.shape[1] != 4096 or err > 1e-4:
        print("    FAIL: basis not 4096-dim or not orthonormal")
        return False
    print("    OK")
    return True


def part2_functional(model_name: str, device: str) -> bool:
    print(f"\n[2] functional ablation + CE DiD on {model_name} (synthetic dirs)")
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    model = WhiteBoxModel(model_name, device=device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    hidden = model.model.config.hidden_size
    rng = np.random.default_rng(0)
    syn = {f: {L: rng.standard_normal(hidden).astype(np.float32) for L in range(n_layers)}
           for f in FOUNDATION_ORDER}
    basis, _, _ = md.build_subspace_basis(syn, kind="probe", n_layers=n_layers)
    layers = sorted(basis)
    res = md.measure_dependency(model, _MORAL, _NEUTRAL, basis, layers, keep_per_text=True)
    model.release()

    print(f"    CE moral   {res['ce']['moral']:.4f} -> {res['ce']['moral_ablated']:.4f} "
          f"(Δ {res['delta_ce']['moral']:+.4f})")
    print(f"    CE neutral {res['ce']['neutral']:.4f} -> {res['ce']['neutral_ablated']:.4f} "
          f"(Δ {res['delta_ce']['neutral']:+.4f})")
    print(f"    score {res['moral_dependency_score']:+.4f}, "
          f"ppl_ratio {res['moral_dependency_ppl_ratio']}")
    ok = True
    if not all(math.isfinite(res["ce"][k]) for k in res["ce"]):
        print("    FAIL: non-finite CE"); ok = False
    if abs(res["delta_ce"]["moral"]) < 1e-6 or abs(res["delta_ce"]["neutral"]) < 1e-6:
        print("    FAIL: ablation had no effect on CE"); ok = False
    if abs(math.exp(res["moral_dependency_score"]) - res["moral_dependency_ppl_ratio"]) > 1e-3:
        print("    FAIL: ppl_ratio != exp(score)"); ok = False
    if len(res["per_text"]["moral_nll"]) != len(_MORAL):
        print("    FAIL: per-text array length mismatch"); ok = False
    print("    OK" if ok else "    FAIL")
    return ok


def part3_empty_control(model_name: str, device: str) -> bool:
    print(f"\n[3] empty-basis control on {model_name} (expect exactly 0)")
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    model = WhiteBoxModel(model_name, device=device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    hidden = model.model.config.hidden_size
    empty = {L: np.zeros((0, hidden), np.float32) for L in range(n_layers)}
    res = md.measure_dependency(model, _MORAL, _NEUTRAL, empty, list(range(n_layers)),
                               keep_per_text=False)
    model.release()
    print(f"    deltas {res['delta_ce']} score {res['moral_dependency_score']}")
    ok = abs(res["moral_dependency_score"]) < 1e-9
    print("    OK" if ok else "    FAIL: empty basis should give 0")
    return ok


def part4_real(npz_path: str) -> bool:
    print("\n[4] real end-to-end on cached OLMo-3 7B base (8 texts, real dirs)")
    if not Path(npz_path).exists():
        print(f"    SKIP: {npz_path} not found"); return True
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    moral, neutral, _ = md.load_probing_texts(40, "v2", split="all", max_texts=8)
    dirs = md.du.load_directions(npz_path)
    model = WhiteBoxModel("allenai/Olmo-3-1025-7B", revision="main",
                          access_tier=AccessTier.WEIGHTS)
    basis, _, _ = md.build_subspace_basis(dirs, kind="probe", n_layers=model.info.n_layers)
    layers = sorted(basis)
    res = md.measure_dependency(model, moral, neutral, basis, layers, keep_per_text=False)
    model.release()
    print(f"    score {res['moral_dependency_score']:+.4f}, "
          f"Δmoral {res['delta_ce']['moral']:+.4f}, Δneutral {res['delta_ce']['neutral']:+.4f}")
    ok = len(layers) == 32 and all(math.isfinite(res["ce"][k]) for k in res["ce"])
    print("    OK" if ok else "    FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Local pre-flight for Sprint 5 dependency code.")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B",
                    help="Small cached model for the functional check.")
    ap.add_argument("--device", default="cpu", help="Device for the small model (cpu is fine).")
    ap.add_argument("--base-npz", default=_BASE_NPZ)
    ap.add_argument("--real", action="store_true",
                    help="Also run the real OLMo-3 7B base path (slow; needs it cached).")
    args = ap.parse_args()

    results = {
        "basis": part1_basis(args.base_npz),
        "functional": part2_functional(args.model, args.device),
        "empty_control": part3_empty_control(args.model, args.device),
    }
    if args.real:
        results["real_7b"] = part4_real(args.base_npz)

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
