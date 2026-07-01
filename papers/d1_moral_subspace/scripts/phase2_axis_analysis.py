#!/usr/bin/env python3
"""Direction 1: rich-subspace rank analysis (numpy, local; no GPU).

Tests whether added sources contribute a DISTINGUISHABLE moral axis beyond Moral Stories'
rank-1-moral signal. Uses the base-extracted per-source mean-diff directions:

  * the axis test -- cos(d_source, d_moral): collinear (~>0.85) => adds mass to the existing
    axis, not a new one; distinguishable (lower) => a new moral axis. The persona axis is the
    non-moral reference (orthogonal to morality).
  * effective MORAL rank -- eff-dim of the set {d_moral, d_fables, d_ethics}: how many
    distinguishable moral axes the sources span (rank>1 = the richness the thesis needs).
  * pooled difference spectrum -- top singular values of [MS_diffs; fable_diffs] vs MS alone,
    to see whether a second coherent (non-content) component emerges.

Reads phase2/base (Moral Stories diffs + d_moral + persona) and phase2/axis (d_fables,
d_ethics, fable diffs) produced by phase2_axis_extract.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
from deepsteer.directions import extraction as du  # noqa: E402

COLLINEAR = 0.85  # |cos| above this => adds mass, not a new axis


def _d(npz_dirs, name, layer):
    v = npz_dirs[name][layer].astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def eff_dim(vecs: list[np.ndarray], thresh=0.9) -> int:
    M = np.stack(vecs)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), thresh)) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Rich-subspace rank analysis.")
    ap.add_argument("--base", default=str(HERE.parent / "outputs" / "phase2" / "base"))
    ap.add_argument("--axis", default=str(HERE.parent / "outputs" / "phase2" / "axis"))
    args = ap.parse_args()

    base, axis = Path(args.base), Path(args.axis)
    layer = int(json.load(open(base / "extract_meta.json"))["match_layer"])

    md = du.load_directions(base / "moral_directions.npz")
    d_moral = _d(md, "moral_stories", layer)
    persona = _d(du.load_directions(base / "persona_direction.npz"), "persona", layer)
    ad = du.load_directions(axis / "axis_directions.npz")
    d_fab = _d(ad, "fables", layer) if "fables" in ad else None
    d_eth = _d(ad, "ethics", layer) if "ethics" in ad else None

    def cos(a, b):
        return round(float(np.dot(a, b)), 4)

    report: dict = {"layer": layer, "collinear_threshold": COLLINEAR, "cosines_vs_d_moral": {}}
    report["cosines_vs_d_moral"]["persona_ref_nonmoral"] = cos(persona, d_moral)
    moral_axes = [d_moral]
    if d_fab is not None:
        c = cos(d_fab, d_moral)
        report["cosines_vs_d_moral"]["fables"] = {
            "cos": c, "verdict": "collinear (no new axis)" if abs(c) >= COLLINEAR
            else "DISTINGUISHABLE -> adds a moral axis"}
        moral_axes.append(d_fab)
    if d_eth is not None:
        c = cos(d_eth, d_moral)
        report["cosines_vs_d_moral"]["ethics"] = {
            "cos": c, "verdict": "collinear (skip full build)" if abs(c) >= COLLINEAR
            else "DISTINGUISHABLE -> do the full ETHICS build"}
    if d_fab is not None and d_eth is not None:
        report["cos_fables_ethics"] = cos(d_fab, d_eth)

    report["effective_moral_rank"] = {
        "sources": ["moral_stories"] + (["fables"] if d_fab is not None else [])
        + (["ethics"] if d_eth is not None else []),
        "n_sources": len(moral_axes) + (1 if d_eth is not None else 0),
        "eff_dim_of_source_directions": eff_dim(
            moral_axes + ([d_eth] if d_eth is not None else [])),
    }

    # pooled difference spectrum: MS-only vs MS + fables
    ms_diffs = np.load(base / "diffs_moral_stories.npz")[f"layer{layer}"].astype(np.float64)
    _, s_ms, _ = np.linalg.svd(ms_diffs, full_matrices=False)
    spec = {"ms_only_top_singvals": np.round(s_ms[:6], 2).tolist()}
    fdiff_p = axis / "axis_diffs_fables.npz"
    if fdiff_p.exists():
        fdiffs = np.load(fdiff_p)[f"layer{layer}"].astype(np.float64)
        pooled = np.concatenate([ms_diffs, fdiffs], axis=0)
        _, s_pool, _ = np.linalg.svd(pooled, full_matrices=False)
        spec["pooled_ms_fables_top_singvals"] = np.round(s_pool[:6], 2).tolist()
    report["difference_spectrum"] = spec

    out = axis / "axis_analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print("=== RICH-SUBSPACE RANK ANALYSIS ===")
    pref = report['cosines_vs_d_moral']['persona_ref_nonmoral']
    print(f"  persona (non-moral ref) cos d_moral: {pref}")
    for k in ("fables", "ethics"):
        if k in report["cosines_vs_d_moral"]:
            r = report["cosines_vs_d_moral"][k]
            print(f"  {k:8} cos d_moral = {r['cos']:+.3f}  -> {r['verdict']}")
    print(f"  effective moral rank of source directions: "
          f"{report['effective_moral_rank']['eff_dim_of_source_directions']} "
          f"(of {report['effective_moral_rank']['n_sources']} sources)")
    print(f"  spectrum: {spec}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
