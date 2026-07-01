#!/usr/bin/env python3
"""Subspace purity check: does V_moral isolate moral content in a given model's space?

The GPT-OSS G3 verdict rests on the persona control (P2 crossed the null but stayed below
persona), so the persona control's validity must be CONFIRMED, not assumed. The discriminating
number is moral/neutral separation accuracy: if V_moral still cleanly classifies moral-vs-neutral,
the subspace is isolating moral content -> persona's high value is general entanglement (all
projections inflated), and refusal-below-persona is genuine orthogonality (benign reading). If
separation degrades, the subspace is partially compromised and the verdict needs a purity caveat.

Reports, per tag: (a) moral/neutral separation acc of the primary moral direction AND the rank-3
span-restricted classifier (on the committed act_sample training pairs), (b) the three source-axis
cross-cosines (are the sources still distinct?), (c) each source direction's cosine to the persona
axis (entanglement). Numpy only; no GPU. Reuses committed extraction artifacts.

  python subspace_purity.py gpt_oss gpt_oss_axis 12
  python subspace_purity.py base axis 16          # OLMo comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
import direction_utils as du  # noqa: E402
from phase2_g3_respec import P2, _ortho  # noqa: E402


def _cos(a, b):
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return round(float(a @ b), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="V_moral purity (separation + axis geometry).")
    ap.add_argument("tag")            # gpt_oss | base | instruct | think
    ap.add_argument("axis_tag")       # gpt_oss_axis | axis | axis_instruct | think_axis
    ap.add_argument("layer", type=int)
    args = ap.parse_args()
    tag, layer = args.tag, args.layer

    md = du.load_directions(P2 / tag / "moral_directions.npz")["moral_stories"][layer]
    ad = du.load_directions(P2 / args.axis_tag / "axis_directions.npz")
    df, de = ad["fables"][layer], ad["ethics"][layer]
    per = du.load_directions(P2 / tag / "persona_direction.npz")["persona"][layer]
    X = np.load(P2 / tag / "act_sample.npz")["X"].astype(np.float64)
    y = np.array([1, 0] * (X.shape[0] // 2))[:X.shape[0]]

    Q = _ortho([md / np.linalg.norm(md), df / np.linalg.norm(df), de / np.linalg.norm(de)])
    d_r3 = Q @ (Q.T @ (X[y == 1].mean(0) - X[y == 0].mean(0)))
    res = {
        "tag": tag, "layer": layer, "n_pairs": int(X.shape[0] // 2),
        "moral_neutral_separation_acc": {
            "d_moral": round(du.transfer_metrics(X, y, md)["acc_midpoint"], 3),
            "rank3_span": round(du.transfer_metrics(X, y, d_r3)["acc_midpoint"], 3)},
        "source_axis_cosines": {
            "moral_fables": _cos(md, df), "moral_ethics": _cos(md, de),
            "fables_ethics": _cos(df, de)},
        "persona_entanglement": {
            "moral": _cos(md, per), "fables": _cos(df, per), "ethics": _cos(de, per)},
        "reading": "clean if separation high (V_moral isolates moral content -> persona valid, "
                   "refusal-below-persona is genuine orthogonality)",
    }
    (P2 / tag / "subspace_purity.json").write_text(json.dumps(res, indent=2))
    s = res["moral_neutral_separation_acc"]
    print(f"=== {tag} purity (layer {layer}, n={res['n_pairs']}) ===")
    print(f"  moral/neutral separation acc: d_moral={s['d_moral']} rank3_span={s['rank3_span']}")
    print(f"  source-axis cosines: {res['source_axis_cosines']}")
    print(f"  persona entanglement: {res['persona_entanglement']}")


if __name__ == "__main__":
    main()
