#!/usr/bin/env python3
"""Phase 1 bootstrap CIs for the two-site headline scalars — LOCAL, no GPU.

Reads the per-prompt headline-layer vectors saved by ``extract_two_site.py`` and
the model's moral/persona subspace, then resamples (with replacement) to put 95%
bootstrap CIs on the Phase-1 headline numbers: per-site moral fraction, the
matched-pooling EOP<->CoT-last cosine + moral asymmetry, and the CoT
last-vs-mean cosine + trace-distribution gap. Decoupled from the GPU run so CIs
are cheap and iterable (the n-discipline: small decision-sentence subsets need
CIs, never bare point estimates).

Each bootstrap rep resamples each site's harmful/harmless rows INDEPENDENTLY and
recomputes the mean-diff direction; cross-site quantities are formed from the
resampled directions (a conservative variability estimate — the pools are not
row-aligned, since CoT excludes incoherent prompts).

Usage:
    python papers/7_reasoning/scripts/bootstrap_two_site.py --key ds_r1_llama8b \
        --vectors  .../ds_r1_llama8b/two_site_headline_vectors.npz \
        --moral-npz .../ds_r1_llama8b/exp1_probe_directions.npz \
        --persona-npz .../ds_r1_llama8b/persona_directions.npz \
        --output   .../ds_r1_llama8b/two_site_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_P5))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import direction_utils as du  # noqa: E402
from moral_dependency import build_subspace_basis  # noqa: E402


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _mft_frac(direction, basis):
    """||proj_S(unit(direction))||^2 for orthonormal basis rows (k, hidden)."""
    c = basis @ _unit(direction)
    return float(c @ c)


def _ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "median": round(float(np.median(a)), 6),
        "ci95_lo": round(float(np.percentile(a, 2.5)), 6),
        "ci95_hi": round(float(np.percentile(a, 97.5)), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 two-site bootstrap CIs (local, no GPU)")
    ap.add_argument("--key", required=True)
    ap.add_argument("--vectors", required=True, help="two_site_headline_vectors.npz")
    ap.add_argument("--moral-npz", required=True)
    ap.add_argument("--moral-kind", default="probe")
    ap.add_argument("--b", type=int, default=2000, help="bootstrap reps.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    z = np.load(args.vectors)
    headline = int(z["headline"])
    sites = {}
    for site in ("eop", "cot_last", "cot_mean"):
        if f"{site}_H" in z.files and f"{site}_S" in z.files:
            sites[site] = (z[f"{site}_H"].astype(np.float64), z[f"{site}_S"].astype(np.float64))

    moral = du.load_directions(args.moral_npz)
    n_layers = 1 + max(L for d in moral.values() for L in d)
    basis_by_layer, _rank, _names = build_subspace_basis(
        moral, kind=args.moral_kind, n_layers=n_layers)
    basis = basis_by_layer[headline]  # (k, hidden)

    def direction(H, S, idxH, idxS):
        return _unit(H[idxH].mean(0) - S[idxS].mean(0))

    rng = np.random.default_rng(args.seed)
    # point estimates
    dirs = {s: direction(H, S, np.arange(len(H)), np.arange(len(S))) for s, (H, S) in sites.items()}
    point = {
        "headline_layer": headline,
        "mft_frac": {s: round(_mft_frac(dirs[s], basis), 6) for s in sites},
        "eop_vs_cotlast_cosine": (round(du.cosine(dirs["eop"], dirs["cot_last"]), 4)
                                  if "eop" in sites and "cot_last" in sites else None),
        "moral_asymmetry": (round(_mft_frac(dirs["cot_last"], basis) - _mft_frac(dirs["eop"], basis), 6)
                            if "eop" in sites and "cot_last" in sites else None),
        "cotlast_vs_cotmean_cosine": (round(du.cosine(dirs["cot_last"], dirs["cot_mean"]), 4)
                                      if "cot_last" in sites and "cot_mean" in sites else None),
        "trace_distribution": (round(_mft_frac(dirs["cot_mean"], basis) - _mft_frac(dirs["cot_last"], basis), 6)
                               if "cot_last" in sites and "cot_mean" in sites else None),
    }

    # bootstrap
    acc = {"mft_frac": {s: [] for s in sites}, "eop_vs_cotlast_cosine": [],
           "moral_asymmetry": [], "cotlast_vs_cotmean_cosine": [], "trace_distribution": []}
    for _ in range(args.b):
        bd, bf = {}, {}
        for s, (H, S) in sites.items():
            iH = rng.integers(0, len(H), len(H))
            iS = rng.integers(0, len(S), len(S))
            bd[s] = direction(H, S, iH, iS)
            bf[s] = _mft_frac(bd[s], basis)
            acc["mft_frac"][s].append(bf[s])
        if "eop" in sites and "cot_last" in sites:
            acc["eop_vs_cotlast_cosine"].append(du.cosine(bd["eop"], bd["cot_last"]))
            acc["moral_asymmetry"].append(bf["cot_last"] - bf["eop"])
        if "cot_last" in sites and "cot_mean" in sites:
            acc["cotlast_vs_cotmean_cosine"].append(du.cosine(bd["cot_last"], bd["cot_mean"]))
            acc["trace_distribution"].append(bf["cot_mean"] - bf["cot_last"])

    ci = {"mft_frac": {s: _ci(acc["mft_frac"][s]) for s in sites}}
    for k in ("eop_vs_cotlast_cosine", "moral_asymmetry",
              "cotlast_vs_cotmean_cosine", "trace_distribution"):
        ci[k] = _ci(acc[k]) if acc[k] else None

    payload = {"analysis": "phase1_two_site_bootstrap", "key": args.key,
               "b": args.b, "sites": list(sites),
               "n_rows": {s: [int(len(H)), int(len(S))] for s, (H, S) in sites.items()},
               "point": point, "bootstrap_ci95": ci}
    Path(args.output).write_text(json.dumps(payload, indent=2))

    print(f"[{args.key}] headline L{headline}  (B={args.b})")
    for s in sites:
        c = ci["mft_frac"][s]
        print(f"  {s:9s} mft_frac {point['mft_frac'][s]:.4f}  CI95 [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]")
    if ci.get("moral_asymmetry"):
        a = ci["moral_asymmetry"]
        print(f"  moral asymmetry (CoT-last - EOP): {point['moral_asymmetry']:+.4f}  "
              f"CI95 [{a['ci95_lo']:+.4f}, {a['ci95_hi']:+.4f}]")
    if ci.get("eop_vs_cotlast_cosine"):
        a = ci["eop_vs_cotlast_cosine"]
        print(f"  EOP<->CoT-last cosine: {point['eop_vs_cotlast_cosine']}  "
              f"CI95 [{a['ci95_lo']:.3f}, {a['ci95_hi']:.3f}]")
    if ci.get("cotlast_vs_cotmean_cosine"):
        a = ci["cotlast_vs_cotmean_cosine"]
        print(f"  CoT last-vs-mean cosine: {point['cotlast_vs_cotmean_cosine']}  "
              f"CI95 [{a['ci95_lo']:.3f}, {a['ci95_hi']:.3f}]")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
