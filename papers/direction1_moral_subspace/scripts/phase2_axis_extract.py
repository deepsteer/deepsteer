#!/usr/bin/env python3
"""Direction 1: extract candidate moral-axis directions on OLMo-3 Base (rich-subspace test).

The single-source V_moral was rank-1-moral + content. To test whether added sources contribute
a DISTINGUISHABLE moral axis (rank>1), extract each candidate source's mean-diff direction +
per-pair difference matrix on the base model at the match layer, so the axis analysis can
measure cos(d_source, d_moral) and the pooled spectrum:

  * fables  -- Understanding Fables clean TRAIN pairs (fables_train_full.json).
  * ethics  -- the committed dataset's eval_generalization_probe (118 clean ETHICS pairs;
               the cosine-collinearity check gates ETHICS's full build).

Saves axis_directions.npz (d per source) + axis_diffs_<source>.npz. Reuses Moral Stories
diffs + d_moral already produced by phase2_extract (base tag). GPU; VALIDATE=1 = tiny smoke.
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
import direction_utils as du  # noqa: E402

MATCH_LAYER = 16
_FULL = HERE.parent / "outputs" / "full"
_DATASET = HERE.parents[2] / "deepsteer" / "datasets" / "direction1_vmoral_v1.json"


def _diff_matrix(X, y):
    Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    return Xn[0::2] - Xn[1::2]


def load_sources() -> dict[str, list[tuple[str, str]]]:
    src: dict[str, list[tuple[str, str]]] = {}
    fab = json.load(open(_FULL / "fables_train_full.json"))
    src["fables"] = [(p["moral"], p["neutral"]) for p in fab["pairs"] if p["clean"]]
    ds = json.load(open(_DATASET))
    src["ethics"] = [(p["moral"], p["neutral"]) for p in ds["eval_generalization_probe"]]
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract candidate moral-axis directions (base).")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "phase2" / "axis"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model = "allenai/OLMo-2-0425-1B"

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    L = min(MATCH_LAYER, model.info.n_layers - 1)

    dirs: dict[str, dict[int, np.ndarray]] = {}
    for label, pairs in sources.items():
        if validate:
            pairs = pairs[:8]
        if not pairs:
            print(f"  {label}: no pairs, skipping")
            continue
        acts = du.collect_pair_activations(model, pairs, input_format="raw", layers=[L])
        dirs[label] = {L: du.mean_diff_direction(*acts[L])}
        np.savez(out / f"axis_diffs_{label}.npz", **{f"layer{L}": _diff_matrix(*acts[L])})
        print(f"  {label}: {len(pairs)} pairs -> d extracted (layer {L})")
    model.release()
    du.save_directions(out / "axis_directions.npz", dirs)
    with open(out / "axis_meta.json", "w") as fh:
        json.dump({"model": args.model, "layer": L, "validate": validate,
                   "n_pairs": {k: len(v) for k, v in sources.items()}}, fh, indent=2)
    print(f"axis extract done -> {out}")


if __name__ == "__main__":
    main()
