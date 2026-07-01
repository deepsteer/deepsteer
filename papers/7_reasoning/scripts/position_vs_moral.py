#!/usr/bin/env python3
"""Phase 2f: position our MFT moral subspace vs Zhao's t_inst harmfulness direction.

LOCAL, no GPU — reuses the moral subspace (Phase 1 exp1_probe_directions.npz) and
the harmfulness direction (Phase 2c position_directions.npz). Computes, per model
at the headline layer:

  * subspace_projection_fraction(harmfulness_t_inst, MFT 6-foundation subspace)
    — fraction of the harmfulness direction lying in the moral subspace;
  * mean |cos| of the harmfulness direction to the 6 foundation directions.

HIGH projection/cosine -> moral-foundations and harmfulness-judgment are the same
object here (our program converges with Zhao; MFT is a structured lens on the same
representation). LOW -> moral-foundations is DISTINCT from harmfulness-judgment
(models route refusal-comprehension through harmfulness, not Haidt's foundations
per se) — its own result and a sharp positioning vs Zhao. Either is publishable.

Usage:
    python papers/7_reasoning/scripts/position_vs_moral.py \
        --outputs-dir papers/7_reasoning/outputs \
        --output papers/7_reasoning/outputs/position_vs_moral.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepsteer.directions import extraction as du  # noqa: E402
import model_registry as reg  # noqa: E402
from moral_dependency import build_subspace_basis  # noqa: E402
from heretic_ablation import subspace_projection_fraction  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402


def analyze(out_dir: Path, key: str) -> dict | None:
    moral_npz = out_dir / key / "exp1_probe_directions.npz"
    pos_npz = out_dir / key / "position_directions.npz"
    if not (moral_npz.exists() and pos_npz.exists()):
        return None
    spec = reg.get(key)
    L = spec.primary_layer
    moral = du.load_directions(str(moral_npz))
    n_layers = 1 + max(Lx for d in moral.values() for Lx in d)
    basis_by_layer, _rank, _names = build_subspace_basis(moral, kind="probe", n_layers=n_layers)
    basis = basis_by_layer[L]                                  # (k, hidden) orthonormal rows
    harm = du.load_directions(str(pos_npz))["harmfulness_t_inst"][L]

    frac = subspace_projection_fraction(harm, [basis[i] for i in range(basis.shape[0])])
    foundations = [f for f in FOUNDATION_ORDER if f in moral and L in moral[f]]
    cos_to_found = {FOUNDATION_SHORT.get(f, f): round(du.cosine(harm, moral[f][L]), 4)
                    for f in foundations}
    mean_abs = round(float(np.mean([abs(v) for v in cos_to_found.values()])), 4) if cos_to_found else None
    # Chance floor: a random direction's norm-ratio into a k-dim subspace ~ sqrt(k/D).
    k, hidden = int(basis.shape[0]), int(harm.shape[0])
    floor = round(float(np.sqrt(k / hidden)), 4)
    return {"headline_layer": L, "moral_subspace_dim": k, "hidden": hidden,
            "harmfulness_in_moral_subspace_fraction": round(float(frac), 4),
            "random_floor_fraction": floor,
            "above_chance_ratio": round(float(frac) / floor, 2),
            "mean_abs_cos_to_foundations": mean_abs, "cos_to_foundations": cos_to_found}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2f MFT-moral vs Zhao-harmfulness positioning")
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--keys", default=",".join(reg.PANEL_ORDER))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.outputs_dir)
    keys = [k.strip() for k in args.keys.split(",")]
    results = {k: r for k in keys if (r := analyze(root, k)) is not None}

    fracs = [r["harmfulness_in_moral_subspace_fraction"] for r in results.values()]
    mean_frac = round(float(np.mean(fracs)), 4) if fracs else None
    # A 6-dim subspace in hidden-dim D captures ~6/D of a random direction; report
    # that floor so "high vs low" is judged against chance, not zero.
    reading = ("converge: harmfulness lies substantially in the MFT moral subspace"
               if (mean_frac or 0) >= 0.5 else
               "distinct: harmfulness is largely OUTSIDE the MFT moral subspace -> "
               "moral-foundations != harmfulness-judgment (sharp positioning vs Zhao)")
    payload = {"analysis": "phase2f_position_vs_moral", "models": results,
               "mean_harmfulness_in_moral_fraction": mean_frac, "reading": reading}
    Path(args.output).write_text(json.dumps(payload, indent=2))

    print("=== Phase 2f: harmfulness(t_inst) vs MFT moral subspace ===")
    print(f"  {'model':14} {'L':>3} {'harm_in_moral':>13} {'rand_floor':>10} {'x_chance':>8} "
          f"{'mean|cos|':>9}")
    for k, r in results.items():
        print(f"  {k:14} {r['headline_layer']:>3} {r['harmfulness_in_moral_subspace_fraction']:>13} "
              f"{r['random_floor_fraction']:>10} {r['above_chance_ratio']:>7}x "
              f"{r['mean_abs_cos_to_foundations']:>9}")
    print(f"\n  mean harmfulness-in-moral fraction: {mean_frac}\n  READING: {reading}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
