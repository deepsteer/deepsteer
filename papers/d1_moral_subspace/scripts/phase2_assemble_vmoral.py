#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 2: assemble V_moral (branches on G-AXIS).

Reads `g_axis_decision.json` and pools ONLY the diff matrices of the sources it names
(two-source on PASS, moral_stories-only on FAIL). SVDs the pooled per-pair difference
vectors and takes the top-r right singular vectors as the orthonormal `V_moral` basis,
where r = UNCENTERED effective dimensionality at variance-threshold 0.90 (PREREGISTRATION
§5 -- the shared moral-salience axis is signal, so no centering; `direction_utils`'s
centered `effective_dimensionality` is deliberately NOT used here).

Writes `v_moral.npz` (basis, eff_dim, sources, layer) -- the artifact the null and G3 read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def uncentered_eff_dim(M: np.ndarray, thresh: float = 0.90) -> tuple[int, np.ndarray, np.ndarray]:
    """Top-r right singular vectors of M (no centering). Returns (r, basis_r, sing_vals)."""
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    r = int(np.searchsorted(explained, thresh)) + 1
    return r, vh[:r], s


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 2: assemble V_moral.")
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parent.parent
                                               / "outputs" / "phase2"))
    args = ap.parse_args()

    out = Path(args.artifacts)
    decision = json.load(open(out / "g_axis_decision.json"))
    meta = json.load(open(out / "extract_meta.json"))
    layer = meta["match_layer"]
    sources = decision["v_moral_sources"]
    key = f"layer{layer}"

    pooled = []
    for s in sources:
        npz = np.load(out / f"diffs_{s}.npz")
        pooled.append(npz[key])
    M = np.concatenate(pooled, axis=0)  # (n_pairs_total, hidden)

    r, basis, sing = uncentered_eff_dim(M, 0.90)
    np.savez(out / "v_moral.npz", basis=basis, eff_dim=r, layer=layer,
             sources=np.array(sources), n_pairs=M.shape[0],
             singular_values=sing[:min(len(sing), 50)])
    print(f"V_moral assembled | branch={decision['decision']} sources={sources} "
          f"| n_pairs={M.shape[0]} | UNCENTERED eff_dim @0.90 = {r} (layer {layer})")
    print(f"  basis shape {basis.shape} (orthonormal rows); singvals[:5]="
          f"{np.round(sing[:5], 3).tolist()}")


if __name__ == "__main__":
    main()
